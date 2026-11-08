from typing import Tuple
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import itertools
import os
from tqdm import tqdm
import h5py
import fastremap
from scipy.ndimage import binary_fill_holes, sum as sum_labels
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects
from skimage.transform import resize
from skimage.filters import gaussian
from scripts_formal.utils.elements_samunetr import SamUNetR
from scripts_formal.utils.elements_data import GridImage, GridImageLite, METAMAP
from scripts_formal.utils.elements_cpdynamics import compute_masks
from scripts_formal.utils.elements_result_io import AnnDataResultStore


class SamProcessor:
    def __init__(
        self,
        **kwargs
    ) -> None:
        """DetectionCellPostProcessor for postprocessing prediction maps and detected cells."""
        self.min_mask_size:int = kwargs.get('min_mask_size', 3000)
        self.min_major_axis_length:int|str = kwargs.get('min_major_axis_length', 'auto')
        if self.min_major_axis_length == 'auto':
            self.min_major_axis_length = np.sqrt(self.min_mask_size / np.pi) * 2
        self.conduct_size_filter:bool = kwargs.get('conduct_size_filter', True)
        self.conduct_mask_filter:bool = kwargs.get('conduct_mask_filter', True)
        self.object_size:int = kwargs.get('object_size', 10)

    def stitch_prediction_map(self, input_image:np.ndarray|GridImage, 
                              image_name:str,
                              predict_instance:bool=True,
                              num_instance_class:int=1,
                              model_2_infer:SamUNetR=None, 
                              save_root:str|None=None,
                              anndata_path:str|None=None,
                              append_to_anndata:bool=False,
                              image_id:str|None=None,
                              run_id:str|None=None,
                              overwrite_image:bool=False,
                              target_stitch_channel:list[int]=[2],
                              stitch_batch_size:int=4,
                              **kwargs):
        """
        Stitch grid predictions into one image and optionally append an AnnData image record.

        Args:
            image_name (str): Name of the image to stitch.
            predict_instance (bool): Whether to predict instance segmentation.
            num_instance_class (int): Total number of instance classes including baseline.
            model_2_infer (SamUNetR|None): Model to use for inference.
            save_root (str|None): Directory for legacy single-image HDF5 output.
            anndata_path (str|None): H5AD file for /uns/luma/images output.
            append_to_anndata (bool): Whether to append image-level results to H5AD.
            image_id (str|None): Stable image key in AnnData. Defaults to image_name.
            run_id (str|None): Optional inference run identifier.
            overwrite_image (bool): Whether to replace an existing AnnData image record.
                The old record is removed only after the new one is fully written.
            target_stitch_channel (list[int]): Channels used for inference and raw-image stitching.
            stitch_batch_size (int): Number of patches processed per model batch.

        Kwargs:
            meta_map (dict): Meta map of the image. (Legacy)
            grid_size (tuple): Size of the grid. (Used for ndarray input)
            grid_target_size (tuple): Target size of the grid. (Used for ndarray input)
            margin_ratio (float): Margin ratio of the grid. (Used for ndarray input)
            rescale (float): Rescale factor of the image. (Used for ndarray input)
            normalize (bool): Whether to normalize the image. (Used for ndarray input)
            input_z_size (int): Input z size of the image. (Used for ndarray input)
            blur_skip (bool|None): Whether to blur the encoder1 skip. None defaults to
                True when the model uses image skip (evaluation), False otherwise.
            object_size (int): Object size for instance segmentation.
            max_cpimage_size (int): Maximum size of the CP image.
            flow_threshold (float): Flow threshold for instance segmentation.
            mask_thres (float): Mask threshold for instance segmentation.
            null_hypothesis_threshold (float): Null hypothesis threshold for type classification.
            scale_ref (float): Scale factor for CP-dynamics Gaussian sigma and default niter.
            blur_maps (bool): Whether to Gaussian-smooth mask and HV maps before dynamics.
            niter (int|None): CP-dynamics follow-flow iterations. None uses int(200 * scale_ref).
        Returns:
            dict: Prediction, GT maps, and optional output paths.
        """

        meta_map = kwargs.get('meta_map', {})
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        run_id = 'current' if run_id is None else run_id
    
        # Prepare the grid image object and derive the stitching geometry.
        if isinstance(input_image, np.ndarray):
            grid_size = kwargs.get('grid_size', (2, 2))
            grid_target_size = kwargs.get('grid_target_size', (1024, 1024))
            margin_ratio = kwargs.get('margin_ratio', 0.1)
            rescale = kwargs.get('rescale', 1.0)
            meta_info = METAMAP.build_meta_info_from_meta_map(image_name, meta_map)
            image_data = GridImageLite(
                input_image,
                grid_size=grid_size,
                grid_target_size=grid_target_size,
                margin_ratio=margin_ratio,
                rescale=rescale,
                meta_info=meta_info,
                image_name=image_name,
                normalize=kwargs.get('normalize', False),
                input_z_size=kwargs.get('input_z_size', 10),
            )
        else:
            image_data = input_image
        image_meta_info = dict(getattr(image_data, 'meta_info', {}))
        image_data.enable_geo_aug = False
        image_data.enable_pix_aug = False
        grid_size = image_data.grid_size
        stitch_info = image_data.stitch_info
        ph, pw, mh, mw = stitch_info['patch_h'], stitch_info['patch_w'], \
            stitch_info['margin_h'], stitch_info['margin_w']
        actual_total_h = (ph - mh) * grid_size[0] + mh
        actual_total_w = (pw - mw) * grid_size[1] + mw
        targeted_stitch_size = stitch_info.get('original_h', actual_total_h), stitch_info.get('original_w', actual_total_w)
        targeted_patch_size = (ph, pw)

        if isinstance(target_stitch_channel, int):
            target_stitch_channel = [target_stitch_channel]
        input_z_size = getattr(model_2_infer, 'input_z_size', kwargs.get('input_z_size', 10)) if model_2_infer is not None else kwargs.get('input_z_size', 10)
        if hasattr(image_data, 'patch_dataset') and hasattr(image_data.patch_dataset, 'configure_3d_input'):
            image_data.patch_dataset.configure_3d_input(target_stitch_channel, input_z_size=input_z_size)
        elif hasattr(image_data, 'configure_3d_input'):
            image_data.configure_3d_input(target_stitch_channel, input_z_size=input_z_size)
        if model_2_infer is not None and hasattr(model_2_infer, 'ensure_input_adapter'):
            model_2_infer.ensure_input_adapter(len(target_stitch_channel))

        # Allocate stitched prediction maps, GT maps, and the pixel-to-embedding indexer.
        res_dict = {
            'stitched_mask_map': torch.zeros((1, actual_total_h, actual_total_w), device=device),
            'stitched_hv_map': torch.zeros((2, actual_total_h, actual_total_w), device=device),
            'stitched_type_cyto_map': torch.zeros((num_instance_class, actual_total_h, actual_total_w), device=device),
            'embedding_indexer': torch.zeros((4, actual_total_h, actual_total_w), device=device, dtype=torch.int32)
        }
        gt_dict = {
            'stitched_gt_inst': torch.zeros((actual_total_h, actual_total_w), device=device, dtype=torch.uint16),
            'stitched_gt_cyto_type': torch.zeros((num_instance_class, actual_total_h, actual_total_w), device=device),
            'stitched_raw_image': torch.zeros((1, actual_total_h, actual_total_w), device=device)
        }

        # Prepare legacy HDF5 output and AnnData image-record output.
        if save_root is not None:
            os.makedirs(os.path.join(save_root, image_name), exist_ok=True)
            output_results_path = os.path.join(save_root, image_name, f"{image_name}_results.hdf5")

        result_store = None
        anndata_file = None
        image_group = None
        patch_embedding_group = None
        if append_to_anndata:
            if anndata_path is None:
                raise ValueError("anndata_path must be provided when append_to_anndata=True.")
            image_id = AnnDataResultStore.sanitize_key(image_name if image_id is None else image_id)
            result_store = AnnDataResultStore(anndata_path)
            anndata_file = h5py.File(anndata_path, 'a')
            image_group = result_store.create_image_group(anndata_file, image_id, overwrite=overwrite_image)
            patch_embedding_group = result_store.require_dict_group(image_group, 'patch_embeddings')
            result_store.write_value(anndata_file['uns']['luma'], 'meta_map', meta_map, overwrite=True)

        # Start model inference patch by patch, streaming embeddings as soon as they are produced.
        model_2_infer = model_2_infer.to(device).eval()
        # Evaluation default: blur encoder1 skip when the model uses image skip.
        blur_skip = kwargs.get('blur_skip')
        if blur_skip is None:
            blur_skip = bool(getattr(model_2_infer, 'use_image_skip', False))
        total_patches = grid_size[0] * grid_size[1]
        all_patch_coords = list(itertools.product(range(grid_size[0]), range(grid_size[1])))
        pbar = tqdm(range(0, total_patches, stitch_batch_size), 
                    desc=f"Stitching patches for {image_name}")
        
        if save_root is not None:
            f = h5py.File(output_results_path, 'w')
            group_embedding_sam = f.create_group('embedding_sam')
            _ = f.create_group('stitched_maps_prediction')
            _ = f.create_group('stitched_maps_gt')
            _ = f.create_group('meta_info_gt')
        
        for batch_start_idx in pbar:
            batch_end_idx = min(batch_start_idx + stitch_batch_size, total_patches)
            current_batch_coords = all_patch_coords[batch_start_idx:batch_end_idx]
            
            # Collect patch tensors for the current model batch.
            batch_images = []
            batch_patch_data = []
            for i, j in current_batch_coords:
                patch_data = image_data.__getitem__(key=(i, j))
                batch_patch_data.append(patch_data)
                selected_image = patch_data['image']
                if selected_image.shape[1] != input_z_size:
                    selected_image = F.interpolate(
                        selected_image.unsqueeze(0),
                        size=(input_z_size, selected_image.shape[-2], selected_image.shape[-1]),
                        mode='trilinear',
                        align_corners=False,
                    ).squeeze(0)
                batch_images.append(selected_image)

            batch_images_tensor = torch.stack(batch_images).to(device)

            with torch.no_grad():
                batch_predicted_res = model_2_infer(batch_images_tensor, blur_skip=blur_skip)
                # Cell encodings use the frozen SAM ViT only (LoRA disabled).
                if model_2_infer.lora_modules:
                    batch_embed_feat = model_2_infer.extract_embed_feat(
                        batch_images_tensor, use_lora=False)
                else:
                    batch_embed_feat = batch_predicted_res['embed_feat']

            # Put each patch prediction back into the stitched coordinate system.
            for batch_item_idx, (i, j) in enumerate(current_batch_coords):
                patch_data = batch_patch_data[batch_item_idx]
                predicted_res = {key: val[batch_item_idx] for key, val in batch_predicted_res.items()}
                patch_embedding = batch_embed_feat[batch_item_idx].squeeze(0).cpu().numpy()
                predicted_res['embed_feat'] = batch_embed_feat[batch_item_idx]
                
                if save_root is not None:
                    group_embedding_sam.create_dataset(
                        name=f"patch_{i}_{j}", 
                        data=patch_embedding,
                        compression="gzip"
                    )
                if append_to_anndata:
                    result_store.write_value(patch_embedding_group, f"patch_{i}_{j}", patch_embedding, overwrite=True)
                
                hmh, hcmh = mh // 2, mh - mh // 2
                hmw, hcmw = mw // 2, mw - mw // 2

                slice_h = slice(hmh, -hcmh if hcmh > 0 else None)
                slice_w = slice(hmw, -hcmw if hcmw > 0 else None)

                target_slice_h = slice((i * (ph - mh) + hmh), ((i + 1) * ph - i * mh - hcmh))
                target_slice_w = slice((j * (pw - mw) + hmw), ((j + 1) * pw - j * mw - hcmw))

                mask_key = 'refined_mask' if 'refined_mask' in predicted_res.keys() else 'raw_mask'
                hv_key = 'refined_hv' if 'refined_hv' in predicted_res.keys() else 'raw_hv'

                res_dict['stitched_mask_map'][
                    :, target_slice_h, target_slice_w] += \
                    TF.resize(F.sigmoid(predicted_res[mask_key]), 
                        interpolation=TF.InterpolationMode.BILINEAR, size=targeted_patch_size
                            )[:, slice_h, slice_w]
                res_dict['stitched_hv_map'][
                    :, target_slice_h, target_slice_w] += \
                    TF.resize(predicted_res[hv_key], 
                        interpolation=TF.InterpolationMode.BILINEAR, size=targeted_patch_size
                            )[:, slice_h, slice_w]
                res_dict['stitched_type_cyto_map'][
                    :, target_slice_h, target_slice_w] += \
                    TF.resize(F.softmax(predicted_res['cyto_type'], dim=0), 
                        interpolation=TF.InterpolationMode.BILINEAR, size=targeted_patch_size
                            )[:, slice_h, slice_w]
                
                raw_image_patch = patch_data['image'].max(dim=1).values
                if raw_image_patch.shape[0] > 1:
                    raw_image_patch = raw_image_patch.mean(dim=0, keepdim=True)
                gt_dict['stitched_raw_image'][
                    :, target_slice_h, target_slice_w] = \
                        TF.resize(raw_image_patch,
                                interpolation=TF.InterpolationMode.BILINEAR, size=targeted_patch_size
                                )[:, slice_h, slice_w]
                    
                if 'instance' in patch_data.keys():
                    patch_instance = patch_data['instance']
                    if patch_instance.ndim == 2:
                        patch_instance = patch_instance.unsqueeze(0)
                    gt_dict['stitched_gt_inst'][
                        target_slice_h, target_slice_w] = \
                            TF.resize(patch_instance,
                                interpolation=TF.InterpolationMode.NEAREST, size=targeted_patch_size
                                )[0, slice_h, slice_w]
                if 'type_cyto_map' in patch_data.keys():
                    patch_type_cyto = patch_data['type_cyto_map']
                    if patch_type_cyto.ndim == 2:
                        patch_type_cyto = patch_type_cyto.unsqueeze(0)
                    gt_dict['stitched_gt_cyto_type'][
                        :, target_slice_h, target_slice_w] = \
                            TF.resize(patch_type_cyto,
                                interpolation=TF.InterpolationMode.NEAREST, size=targeted_patch_size
                                )[:, slice_h, slice_w]
                res_dict['embedding_indexer'][0, target_slice_h, target_slice_w] = i
                res_dict['embedding_indexer'][1, target_slice_h, target_slice_w] = j
                res_dict['embedding_indexer'][2:4, target_slice_h, target_slice_w] = \
                    TF.resize(
                        torch.stack(
                        list(torch.meshgrid(
                            torch.arange(predicted_res['embed_feat'].shape[1]), 
                            torch.arange(predicted_res['embed_feat'].shape[2]),
                            indexing='ij'
                        )), dim=0).squeeze(0).to(device),
                        interpolation=TF.InterpolationMode.NEAREST, size=targeted_patch_size
                    )[:, slice_h, slice_w]

        if save_root is not None:
            f.close()

        # Resize maps from the stitched working size back to the original image size.
        for key in ['stitched_mask_map', 'stitched_type_cyto_map']:
            res_dict[key] = TF.resize(
                res_dict[key], size=targeted_stitch_size,
                interpolation=TF.InterpolationMode.BILINEAR)
        for key in ['embedding_indexer']:
            res_dict[key] = TF.resize(
                res_dict[key], size=targeted_stitch_size,
                interpolation=TF.InterpolationMode.NEAREST)
        for key in ['stitched_hv_map']:
            res_dict[key] = TF.resize(
                res_dict[key], size=targeted_stitch_size,
                interpolation=TF.InterpolationMode.BILINEAR)
        
        # Convert class probability maps into a final type index map.
        type_cyto_distri = res_dict['stitched_type_cyto_map']
        type_cyto_distri[0, :, :][type_cyto_distri[0, :, :] < kwargs.get('null_hypothesis_threshold', 0.15)] = kwargs.get('null_hypothesis_threshold', 0.15)
        type_cyto_ind = torch.argmax(type_cyto_distri, dim=0).cpu().numpy()

        res_dict['stitched_mask_map'] = res_dict['stitched_mask_map'].clamp(0, 1).cpu().numpy()
        res_dict['stitched_hv_map'] = res_dict['stitched_hv_map'].cpu().numpy()

        gt_dict['stitched_gt_inst'] = TF.resize(
            gt_dict['stitched_gt_inst'].unsqueeze(0), size=targeted_stitch_size,
            interpolation=TF.InterpolationMode.NEAREST).squeeze(0).cpu().numpy().astype(np.uint16)
        gt_dict['stitched_gt_cyto_type'] = TF.resize(
            gt_dict['stitched_gt_cyto_type'], size=targeted_stitch_size,
            interpolation=TF.InterpolationMode.NEAREST)
        gt_dict['stitched_raw_image'] = TF.resize(
            gt_dict['stitched_raw_image'], size=targeted_stitch_size,
            interpolation=TF.InterpolationMode.NEAREST).squeeze(0).cpu().numpy()
        
        type_cyto_distri_gt = gt_dict['stitched_gt_cyto_type']
        type_cyto_ind_gt = torch.argmax(type_cyto_distri_gt, dim=0).cpu().numpy()
        mask = res_dict['stitched_mask_map'][0]
        res_dict['stitched_hv_map'] = np.stack([
            res_dict['stitched_hv_map'][0],
            res_dict['stitched_hv_map'][1]
        ], axis=0)
        
        prediction_dict = {
            'mask': mask.astype('float32'),
            'h_map': res_dict['stitched_hv_map'][0].astype('float32'),
            'v_map': res_dict['stitched_hv_map'][1].astype('float32'),
            'pred_type_cyto_map': type_cyto_ind.astype('float32'),
            'image_name': image_name,
            'n_types': num_instance_class
        }

        grd_dict = {
            'gt_instance_map': gt_dict['stitched_gt_inst'],
            'gt_type_cyto_map': type_cyto_ind_gt.astype('float32'),
            'gt_meta': {},
            'gt_raw_image': gt_dict['stitched_raw_image']
        }

        # Generate final instances on the stitched prediction maps.
        if predict_instance:
            instance_map = self.get_inst_cpdynamics(
                np.stack([
                    prediction_dict['mask'],
                    prediction_dict['h_map'],
                    prediction_dict['v_map']
                    ], axis=-1),
                object_size=kwargs.get('object_size', self.object_size),
                use_gpu=(device.type == 'cuda'),
                max_size=kwargs.get('max_cpimage_size', 2048),
                flow_threshold=kwargs.get('flow_threshold', 0.5),
                mask_threshold=kwargs.get('mask_thres', 0.5),
                scale_ref=kwargs.get('scale_ref', 1.0),
                blur_maps=kwargs.get('blur_maps', True),
                niter=kwargs.get('niter', None),
            )

            if self.conduct_size_filter:
                instance_map = self.filter_inst(instance_map, min_major_axis_length=self.min_major_axis_length)

            prediction_dict['pred_instance_map'] = instance_map
        
        # Preserve the old single-image HDF5 output path for compatibility.
        if save_root is not None:
            with h5py.File(output_results_path, 'a') as f:
                pred_group_map = f['stitched_maps_prediction']
                gt_group_map = f['stitched_maps_gt']
                gt_group_meta = f['meta_info_gt']
                for key in prediction_dict.keys():
                    if key not in ['image_name', 'n_types']:
                        pred_group_map.create_dataset(
                            name=key,
                            data=prediction_dict[key],
                            compression="gzip"
                        )
                for key in grd_dict.keys():
                    if key not in ['gt_meta']:
                        gt_group_map.create_dataset(
                            name=key,
                            data=grd_dict[key],
                            compression="gzip"
                        )
                for k, v in grd_dict['gt_meta'].items():
                    gt_group_meta.create_dataset(
                        name=k,
                        data=v,
                        compression="gzip"
                    )
                f.create_dataset(
                    name="embedding_indexer",
                    data=res_dict['embedding_indexer'].cpu().numpy(),
                    compression="gzip"
                )
                f.attrs['image_name'] = prediction_dict['image_name']
                f.attrs['n_types'] = prediction_dict['n_types']
                f.attrs['meta_map'] = str(meta_map)
                f.attrs['stitch_info'] = str(stitch_info)
            prediction_dict['total_result_path'] = output_results_path

        # Append the AnnData image record after all stitched maps are finalized.
        if append_to_anndata:
            result_bundle = {
                'attrs': {
                    'image_id': image_id,
                    'run_id': run_id,
                    'image_name': image_name,
                    'n_types': num_instance_class,
                    'grid_rows': grid_size[0],
                    'grid_cols': grid_size[1],
                    'meta_info': image_meta_info,
                },
                'prediction': {k: v for k, v in prediction_dict.items() if k not in ['image_name', 'n_types', 'total_result_path']},
                'gt': {
                    'gt_instance_map': grd_dict['gt_instance_map'],
                    'gt_type_cyto_map': grd_dict['gt_type_cyto_map'],
                    'gt_raw_image': grd_dict['gt_raw_image'],
                },
                'embedding_indexer': res_dict['embedding_indexer'].cpu().numpy(),
                'stitch_info': stitch_info,
                'postprocess_params': {
                    'predict_instance': predict_instance,
                    'target_stitch_channel': target_stitch_channel,
                    'mask_thres': kwargs.get('mask_thres', 0.5),
                    'flow_threshold': kwargs.get('flow_threshold', 0.5),
                    'null_hypothesis_threshold': kwargs.get('null_hypothesis_threshold', 0.15),
                    'max_cpimage_size': kwargs.get('max_cpimage_size', 2048),
                    'scale_ref': kwargs.get('scale_ref', 1.0),
                    'blur_maps': kwargs.get('blur_maps', True),
                    'niter': kwargs.get('niter', None),
                },
                'model_info': {
                    'model_class': type(model_2_infer).__name__,
                },
            }
            # Write the record under a staging key and replace the final record only after all fields are written.
            try:
                result_store.write_image_result_to_group(image_group, result_bundle)
                result_store.commit_image_group(anndata_file, image_id, overwrite=overwrite_image)
            except Exception:
                result_store.discard_image_group(anndata_file, image_id)
                raise
            finally:
                anndata_file.close()
            prediction_dict['anndata_path'] = anndata_path
            prediction_dict['image_id'] = image_id

        prediction_dict.update(grd_dict)
        return prediction_dict

    def predict_single_image(self, 
                             input_image: np.ndarray, 
                             model_2_infer: SamUNetR,
                             target_size: Tuple[int, int] = (1024, 1024),
                             max_cpimage_size: int = 256,
                             flow_threshold: float = 0.4,
                             blur_skip: bool | None = None,
                             **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Predict instance segmentation and type map for a single image.

        Args:
            input_image: 2D grayscale array. Values in 0–255 are divided by 255.
            model_2_infer: SamUNetR used for one-FOV inference.
            target_size: Spatial size fed to the model, ``(H, W)``.
            max_cpimage_size: Long-side canvas for CP-dynamics. Historical default is 256.
            flow_threshold: Cellpose follow-flow threshold passed to ``get_inst_cpdynamics``.
            blur_skip: Whether to blur the encoder1 skip. ``None`` follows
                ``model_2_infer.use_image_skip``.
        """
        device = next(model_2_infer.parameters()).device
        original_shape = input_image.shape
        
        # Resize and normalize one input image before model inference.
        if original_shape != target_size:
            img_resized = cv2.resize(input_image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = input_image
            
        img_resized = img_resized.astype(np.float32)
        img_resized = img_resized / 255
        img_tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).unsqueeze(2).to(device)
        if hasattr(model_2_infer, 'ensure_input_adapter'):
            model_2_infer.ensure_input_adapter(image_in_channels=1)
        
        # Run the segmentation model and collect raw maps.
        # Evaluation default: blur encoder1 skip when the model uses image skip.
        if blur_skip is None:
            blur_skip = bool(getattr(model_2_infer, 'use_image_skip', False))
        model_2_infer.eval()
        with torch.no_grad():
            res = model_2_infer(img_tensor, blur_skip=blur_skip)
            
        if 'refined_mask' in res.keys():
            mask_pred = F.sigmoid(res['refined_mask'].squeeze().cpu()).numpy()
        else:
            mask_pred = F.sigmoid(res['raw_mask'].squeeze().cpu()).numpy()
        if 'refined_hv' in res.keys():
            hv_pred = res['refined_hv'].squeeze().cpu().numpy()
        else:
            hv_pred = res['raw_hv'].squeeze().cpu().numpy()

        stack_in = np.stack([mask_pred, hv_pred[0], hv_pred[1]], axis=-1)
        type_logits = F.sigmoid(res['cyto_type'].squeeze(0))
        type_probs = F.softmax(type_logits, dim=0).cpu().numpy()
        type_probs_resized = np.zeros((type_probs.shape[0], 256, 256), dtype=type_probs.dtype)
        for i in range(type_probs.shape[0]):
            type_probs_resized[i] = gaussian(resize(type_probs[i], (256, 256)), sigma=1)
        type_pred = np.argmax(type_probs_resized, axis=0).astype(np.float32)
        
        # Convert maps into instances, then resize them back to the original image.
        inst_map = self.get_inst_cpdynamics(
            stack_in,
            use_gpu=(device.type == 'cuda'),
            max_size=max_cpimage_size,
            flow_threshold=flow_threshold,
        )

        if original_shape != (256, 256):
            inst_map = cv2.resize(inst_map.astype(np.uint16), (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
        type_pred = cv2.resize(type_pred.astype(np.float32), (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)

        if self.conduct_size_filter:
            inst_map = self.filter_inst(inst_map, min_major_axis_length=self.min_major_axis_length)
        
        # Assign an instance-level type by majority vote on the pixel-level type map.
        null_hypothesis_threshold = kwargs.get('null_hypothesis_threshold', 0.8)
        all_inst_labels = np.unique(inst_map)
        all_inst_labels = all_inst_labels[all_inst_labels > 0]
        max_inst_id = np.max(inst_map) if len(all_inst_labels) > 0 else 0
        pred_lut = np.zeros(max_inst_id + 1, dtype=np.int64)
        
        if len(all_inst_labels) > 0:
            n_classes = int(np.max(type_pred)) + 1
            counts_list = []
            type_pred_int = type_pred.astype(np.int32)
            for i in range(n_classes):
                class_mask = (type_pred_int == i).astype(np.int32)
                c = sum_labels(class_mask, labels=inst_map, index=all_inst_labels)
                counts_list.append(c)
            counts_matrix = np.vstack(counts_list).T
            total_pixels = counts_matrix.sum(axis=1)
            scores = np.zeros_like(counts_matrix, dtype=np.float32)
            valid_mask = total_pixels > 0
            scores[valid_mask] = counts_matrix[valid_mask] / total_pixels[valid_mask, None]
            is_background = scores[:, 0] > null_hypothesis_threshold
            if n_classes > 1:
                best_non_bg = np.argmax(scores[:, 1:], axis=1) + 1
            else:
                best_non_bg = np.zeros(len(all_inst_labels), dtype=np.int64)
            final_types = np.where(is_background, 0, best_non_bg)
            if kwargs.get('type_idx_shift', 0):
                final_types += kwargs.get('type_idx_shift')
            pred_lut[all_inst_labels] = final_types

        instance_type_map = pred_lut[inst_map]
        return inst_map.astype(np.uint16), type_pred, instance_type_map

    def get_inst_cpdynamics(self, pred_inst, object_size:int=10, max_size:int=2048,
                            flow_threshold:float=1.0, mask_threshold:float=0.5,
                            use_gpu:bool=False, precision=torch.double,
                            scale_ref=1.0, blur_maps:bool=True,
                            niter:int|None=None) -> np.ndarray:
        """Convert mask and HV predictions into an instance map with CP-dynamics.

        Defaults match the historical path: Gaussian-smooth maps with sigma=1*scale_ref
        and run int(200*scale_ref) follow-flow iterations. The mask is smoothed in 2D;
        each HV component is smoothed separately over its own spatial axes.
        """

        mask = pred_inst[..., 0]
        dP = np.stack([pred_inst[..., 2], pred_inst[..., 1]], axis=0)
        original_shape = mask.shape
        niter_eff = int(200 * scale_ref) if niter is None else int(niter)

        # Downsample large maps before CP-dynamics to keep postprocessing manageable.
        if max_size is not None:
            scale = max_size / max(original_shape[0], original_shape[1])
            # if scale < 0.25:
            #     scale = 0.25
            Ly, Lx = int(original_shape[0] * scale), int(original_shape[1] * scale)
            dP_resized = np.zeros((2, Ly, Lx), dtype=dP.dtype)
            dP_resized[0] = cv2.resize(dP[0], (Lx, Ly), interpolation=cv2.INTER_LINEAR)
            dP_resized[1] = cv2.resize(dP[1], (Lx, Ly), interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(mask, (Lx, Ly), interpolation=cv2.INTER_LINEAR)

            if blur_maps:
                mask_resized = gaussian(mask_resized, sigma=1 * scale_ref)
                # Smooth each flow component over spatial axes only; smoothing the (2, H, W)
                # stack as a whole would mix the horizontal and vertical components.
                dP_resized = np.stack([
                    gaussian(dP_resized[0], sigma=1 * scale_ref),
                    gaussian(dP_resized[1], sigma=1 * scale_ref),
                ], axis=0)

            if self.conduct_mask_filter:
                thred_mask = remove_small_objects(mask_resized > mask_threshold,
                    min_size=int(self.min_mask_size * scale * scale)).astype('float32')
            else:
                thred_mask = (mask_resized > mask_threshold).astype('float32')
            thred_mask = binary_fill_holes(thred_mask > 0).astype('float32')
            dP_resized = dP_resized * np.expand_dims(thred_mask, axis=0)
            inst, _ = compute_masks(dP_resized, mask_resized,
                                    niter=niter_eff,
                                    cellprob_threshold=mask_threshold,
                                    flow_threshold=flow_threshold,
                                    min_size=object_size,
                                    resize=original_shape,
                                    use_gpu=use_gpu,
                                    precision=precision
                                    )
        else:
            dP = dP * np.expand_dims((mask > mask_threshold).astype('float32'), axis=0)
            inst, _ = compute_masks(dP, mask,
                                    niter=niter_eff,
                                    cellprob_threshold=mask_threshold,
                                    flow_threshold=flow_threshold,
                                    min_size=object_size,
                                    use_gpu=use_gpu,
                                    precision=precision
                                    )

        return inst.astype(np.uint16)

    @staticmethod
    def filter_inst(instance_map, min_major_axis_length:int=0):
        """Filter instances based on minimum major-axis length and relabel the map."""
        properties = regionprops(instance_map)
        convert_lut = np.arange(np.max(instance_map) + 1, dtype=np.uint16)
        
        for prop in properties:
            if prop.major_axis_length < min_major_axis_length:
                convert_lut[prop.label] = 0
        
        filtered_instance_map = convert_lut[instance_map]
        relabeled_map = np.array(filtered_instance_map, copy=True)
        fastremap.renumber(relabeled_map, in_place=True)
        return relabeled_map
