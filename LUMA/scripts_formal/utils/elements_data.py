import os
import cv2
import numpy as np
import random
import warnings
from tqdm import tqdm
import pickle as pkl
import h5py
from pathlib import Path
from typing import List, Tuple, Literal
from skimage.measure import regionprops
from skimage.measure._regionprops import RegionProperties
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.segmentation import relabel_sequential
from scipy.ndimage import maximum_filter, minimum_filter, sum_labels
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from collections.abc import Callable
from ast import literal_eval
from scripts_formal.utils.elements_cpdynamics import generate_patch_hvmaps_2d
from scripts_formal.utils.metamap_function_pairs import METAMAP_FUNCTION_PAIRS


class METAMAP:
    """Encode image_name as a metadata value or decode that value into a filename keyword.

    Dataset-specific encoding and decoding live in metamap_function_pairs.py; this class only wraps them.
    """

    def __init__(self, ffov, finv):
        """Store a pair of encoding and decoding functions."""
        self.ffov = ffov
        self.finv = finv

    def encode(self, image_name):
        """Encode an image name as a metadata value."""
        return self.ffov(image_name)

    def decode(self, meta_value):
        """Decode a metadata value into the dataset's time keyword."""
        return self.finv(meta_value)

    @staticmethod
    def build_meta_info_from_meta_map(image_name, meta_map):
        """Generate image-level metadata using a dict mapper or a registered METAMAP name."""
        meta_info = {}

        for meta_key, mapper in meta_map.items():
            # Assign sequential indexes during dataset construction; mutate mapper in place.
            if meta_key == 'batch_name':
                batch_idx = len(mapper)
                mapper[image_name] = batch_idx
                meta_info[meta_key] = batch_idx
            # Substring dictionary: select the longest key matching image_name.
            elif isinstance(mapper, dict):
                hitted_key = None
                for key in mapper.keys():
                    if key in image_name:
                        if hitted_key is None or len(key) > len(hitted_key):
                            hitted_key = key
                meta_info[meta_key] = mapper[hitted_key]
            # Registered name: call the corresponding encode function.
            elif isinstance(mapper, str) and mapper.startswith('METAMAP_'):
                meta_info[meta_key] = METAMAP_REGISTRY[mapper].encode(image_name)
            else:
                raise TypeError(f"Unsupported mapper for meta key '{meta_key}': {mapper}")

        return meta_info

    @staticmethod
    def decode_mapper_value(mapper, meta_value):
        """Decode an encoded value into a name keyword using one meta_map entry."""
        if isinstance(mapper, dict):
            hitted_key = None
            for key, value in mapper.items():
                if value != meta_value:
                    continue
                if hitted_key is None or len(str(key)) > len(str(hitted_key)):
                    hitted_key = key
            return "" if hitted_key is None else str(hitted_key)
        if isinstance(mapper, str) and mapper.startswith('METAMAP_'):
            return str(METAMAP_REGISTRY[mapper].decode(meta_value))
        return str(meta_value)


METAMAP_REGISTRY = {
    name: METAMAP(ffov, finv)
    for name, (ffov, finv) in METAMAP_FUNCTION_PAIRS.items()
}


class GridImageDataset:
    def __init__(self, 
                 data_storage_path:Path|str,
                 meta_map:dict={}):

        self.grid_image = {}
        self.merged_dataset = {'patch': None, 'sc': None}
        self.train_test_names = {'train': [], 'test': []}
        self.data_storage_path = Path(data_storage_path)
        self.hdf5_path = self.data_storage_path / 'data.h5'
        self.metadata_path = self.data_storage_path / 'metadata.pkl'
        self.meta_map:dict = meta_map
        self.meta_map.update({'batch_name': {}})
        os.makedirs(self.data_storage_path, exist_ok=True)
        
        self.default_loading_option = {
            'grid_size': (2, 2),
            'grid_target_size': (1024, 1024),
            'field_target_size': (1536, 1536),
            'margin_ratio': 0.1,
            'rescale': 1.0,
            'generate_classification': {},
            'type_mode': 'inst_wise', # 'inst_wise' or 'signal_wise'
            'cell_target_size': (224, 224), 
            'extension_rate': 1.0,
            'object_attri_limit': {'axis_major_length': (85, None)},  # {'attri_name': (lower_limit, upper_limit)}
            'defined_attri_func': None,  # Described as extra properties for regionprops
            'normalize': False,
            'delete_edges': False,
            'input_z_size': 10,
            'image_axis_order': 'auto'
        }

    @property
    def train_names(self) -> list[str]:
        return self.train_test_names.get('train', [])

    @property
    def test_names(self) -> list[str]:
        return self.train_test_names.get('test', [])

    def configure_default_loading_option(self, **kwargs):
        """Configure default loading options for grid images.

        Args:
            **kwargs: Keyword arguments for loading options. See in load_grid_image for details.
        """
        for key in kwargs:
            if key in self.default_loading_option:
                self.default_loading_option[key] = kwargs[key]
            else:
                raise KeyError(f"Invalid loading option: {key}")

    def update_storage_path(self, new_data_storage_path: Path|str):
        """Update the data storage path for the dataset and all its components.

        Args:
            new_data_storage_path (Path|str): The new path to the data storage directory.
        """
        self.data_storage_path = Path(new_data_storage_path)
        self.hdf5_path = self.data_storage_path / 'data.h5'
        self.metadata_path = self.data_storage_path / 'metadata.pkl'
        assert self.data_storage_path.exists(), f"New data storage path {self.data_storage_path} does not exist."
        assert self.hdf5_path.exists(), f"HDF5 file not found at {self.hdf5_path}."

        for grid_img in self.grid_image.values():
            grid_img.hdf5_path = self.hdf5_path
            if grid_img.patch_dataset is not None:
                grid_img.patch_dataset.hdf5_path = self.hdf5_path
            if grid_img.sc_dataset is not None:
                grid_img.sc_dataset.hdf5_path = self.hdf5_path

        if self.merged_dataset['patch'] is not None:
            self.merged_dataset['patch'].hdf5_path = self.hdf5_path
        if self.merged_dataset['sc'] is not None:
            self.merged_dataset['sc'].hdf5_path = self.hdf5_path

        self.save()

    def get_dataset(
        self,
        image_path_template: str|list[str], 
        mask_path_template: str|list[str],
        path_kw: list[tuple[str]|str],
        mask_kw: list[tuple[str]|str]|None = None,
        image_names: list[str]|None = None,
        **kwargs
    ):  
        """Generate a GridImageDataset from the given image and mask paths.

        Args:
            image_path_template (str|list[str]): Template for image paths.
            mask_path_template (str|list[str]): Template for mask paths.
            path_kw (list[tuple[str]|str]): Keyword arguments for image and mask paths.
            mask_kw (list[tuple[str]|str]|None): Keyword arguments for mask paths, if different from path_kw.
            image_names (list[str]|None): Optional list of image names.
            **kwargs: Additional keyword arguments for loading images. See in load_grid_image for details.

        Returns:
            GridImageDataset: An instance of GridImageDataset containing the loaded images and masks.
        """

        mask_path_template = [mask_path_template] if isinstance(image_path_template, str) else mask_path_template
        image_path_template = [image_path_template] if isinstance(image_path_template, str) else image_path_template
        mask_kw = path_kw if mask_kw is None else mask_kw
        assert len(mask_kw) == len(path_kw), \
            f"Length of mask_kw ({len(mask_kw)}) must match length of path_kw ({len(path_kw)})"

        for idx, kw in enumerate(zip(path_kw, mask_kw)):
            print(f"Loading image and mask set {image_names[idx] if image_names is not None else idx+1}")
            img_kw = kw[0] if isinstance(kw[0], tuple) else (kw[0],)
            mask_kw = kw[1] if isinstance(kw[1], tuple) else (kw[1],)

            self.load_grid_image(
                image_paths=[temp.format(*img_kw) for temp in image_path_template],
                mask_path=[temp.format(*mask_kw) for temp in mask_path_template],
                image_name=image_names[idx] if image_names is not None else f"grid_image_{idx+1}",
                **kwargs)

        return

    def load_grid_image(self,
                        image_paths:str|Path|List[str]|List[Path], 
                        mask_path:str|Path|List[str]|List[Path],
                        image_name:str|None=None,
                        **kwargs
                        ) -> None: 
        """Load a GridImageDataset from the provided image and mask paths.

        Args:
            image_paths (str|Path|List[str]|List[Path]): Path(s) to the image(s) to be loaded.
            mask_path (str|Path|List[str]|List[Path]): Path to the mask corresponding to the images.
            image_name (str|None): Optional name for the grid image. If None, a default name will be generated.
            **kwargs: Additional keyword arguments for grid size, target size, margin ratio, rescale, align mode, etc.
                * grid_size (Tuple[int, int]|int): Size of the grid. If int, will create a square grid.
                * grid_target_size (Tuple[int, int]|int): Target size of each grid patch. If int, will create a square patch.
                * field_target_size (Tuple[int, int]|int): Target size of the whole field of view after resizing, used to determine grid_size and margin_ratio when grid_size is set to 'auto'. If int, will create a square size.
                * margin_ratio (float|tuple|list): Ratio of the margin to be applied to each patch.
                * rescale (float): Rescale factor for the images.
                * generate_classification (dict): If provided, will generate classification labels for each patch.
                    * e.g. of generate_classification: {'channel': [int], 'threshold': [float|Literal['auto']]}, where 'channel' is the channel index to use for classification,
                    and 'threshold' is the threshold value for classification. If 'auto', will use Otsu's method to determine the threshold.
                * type_mode (Literal['inst_wise', 'signal_wise']): Mode for generating type maps.
                * cell_target_size (Tuple[int, int]|int): Target size for single-cell images. If int, will create a square size.
                * extension_rate (float): Extension rate for cell bounding boxes when extracting single-cell images.
                * object_attri_limit (dict|None): Dictionary containing attribute limits for objects in the mask.
                    * e.g. of object_attri_limit: {'area': (None, 1000), 'axis_major_length': (None, 100)}.
                * defined_attri_func (list[Callable]|None): List of functions to be used as extra properties for regionprops.
                * normalize (bool): Whether to normalize the images.
                * delete_edges (bool): Whether to delete edge objects when dealing with mask image.

        """

        if image_name is None:
            image_name = f"grid_image_{len(list(self.grid_image.keys())) + 1}"
        assert image_name not in list(self.grid_image.keys()), \
            f"Grid image with name '{image_name}' already exists. Please choose a different name."
        
        meta_info = METAMAP.build_meta_info_from_meta_map(image_name, self.meta_map)
            
        self.grid_image[image_name] = GridImage(
            hdf5_path=self.hdf5_path,
            group_name=image_name,
            image_paths=image_paths,
            mask_path=mask_path,
            meta_info=meta_info,
            grid_size=kwargs.get('grid_size', self.default_loading_option['grid_size']),
            grid_target_size=kwargs.get('grid_target_size', self.default_loading_option['grid_target_size']),
            field_target_size=kwargs.get('field_target_size', self.default_loading_option['field_target_size']),
            margin_ratio=kwargs.get('margin_ratio', self.default_loading_option['margin_ratio']),
            rescale=kwargs.get('rescale', self.default_loading_option['rescale']),
            generate_classification=kwargs.get('generate_classification', self.default_loading_option['generate_classification']),
            type_mode=kwargs.get('type_mode', self.default_loading_option['type_mode']),
            cell_target_size=kwargs.get('cell_target_size', self.default_loading_option['cell_target_size']),
            extension_rate=kwargs.get('extension_rate', self.default_loading_option['extension_rate']),
            object_attri_limit=kwargs.get('object_attri_limit', self.default_loading_option['object_attri_limit']),
            defined_attri_func=kwargs.get('defined_attri_func', self.default_loading_option['defined_attri_func']),
            normalize=kwargs.get('normalize', self.default_loading_option['normalize']),
            delete_edges=kwargs.get('delete_edges', self.default_loading_option['delete_edges']),
            input_z_size=kwargs.get('input_z_size', self.default_loading_option['input_z_size']),
            image_axis_order=kwargs.get('image_axis_order', self.default_loading_option['image_axis_order'])
        )

    def get_merged_dataset(self,
                           image_name_list:List[str]):
        """Get a merged dataset from multiple grid images."""

        train_name_set = set(image_name_list)
        self.train_test_names = {
            'train': image_name_list,
            'test': [name for name in list(self.grid_image.keys()) if name not in train_name_set]
        }

        if len(image_name_list) == 0:
            self.merged_dataset['patch'] = None
            self.merged_dataset['sc'] = None
            return None, None

        # Check if all image names are in the grid_image dataset
        assert all(name in self.grid_image for name in image_name_list), \
            "Some image names are not found in the grid_image dataset."
        
        # Check if all output tensors have the same shape
        if len(image_name_list) > 0:
            first_name = image_name_list[0]
            first_ds = self.grid_image[first_name].patch_dataset
            first_t_size = first_ds.target_size
            first_type_num = first_ds.type_num
            
            for name in image_name_list[1:]:
                current_ds = self.grid_image[name].patch_dataset
                assert current_ds.target_size == first_t_size, \
                    f"Target size mismatch: {name} has {current_ds.target_size}, expected {first_t_size}"
                assert current_ds.type_num == first_type_num, \
                    f"Type num mismatch: {name} has {current_ds.type_num}, expected {first_type_num}"
        
        # Get merged dataset
        merged_patch_dataset = GridDataSet.merge_dataset([self.grid_image[name].patch_dataset for name in image_name_list])
        merged_sc_dataset = scDataSet.merge_dataset([self.grid_image[name].sc_dataset for name in image_name_list])
        self.merged_dataset['patch'] = merged_patch_dataset
        self.merged_dataset['sc'] = merged_sc_dataset

        return merged_patch_dataset, merged_sc_dataset

    def save(self):
        """Save the GridImageDataset object state to a pickle file."""
        with open(self.metadata_path, 'wb') as f:
            pkl.dump(self, f)

    @classmethod
    def load(cls, data_storage_path:Path|str) -> "GridImageDataset":
        """Load a GridImageDataset from a directory."""
        data_storage_path = Path(data_storage_path)
        metadata_path = data_storage_path / 'metadata.pkl'

        if not metadata_path.exists():
            raise ValueError(f"Metadata file not found in {data_storage_path}")
        
        with open(metadata_path, 'rb') as f:
            gid = pkl.load(f)

        gid.data_storage_path = Path(gid.data_storage_path)
        gid.hdf5_path = Path(gid.hdf5_path)
        gid.metadata_path = Path(gid.metadata_path)

        if gid.data_storage_path != data_storage_path:
            gid.update_storage_path(data_storage_path)
        
        # pickle already stores patch_info / cell_info; rebuilding them here
        # re-opens data.h5 once per image and is not needed after update_storage_path.
        # for gi in gid.grid_image.values():
        #     if hasattr(gi, 'patch_dataset') and isinstance(gi.patch_dataset, GridDataSet):
        #         gi.patch_dataset._get_patch_info()
        #     if hasattr(gi, 'sc_dataset') and isinstance(gi.sc_dataset, scDataSet):
        #         gi.sc_dataset._get_cell_info()
        # if gid.merged_dataset['patch'] is not None:
        #     gid.merged_dataset['patch']._get_patch_info()
        # if gid.merged_dataset['sc'] is not None:
        #     gid.merged_dataset['sc']._get_cell_info()

        return gid

    @staticmethod
    def _rebuild_train_test_names(grid_image: dict, merged_dataset) -> dict[str, list[str]]:
        all_image_names = list(grid_image.keys())
        merged_group_paths = []

        if merged_dataset is not None and hasattr(merged_dataset, 'group_paths'):
            merged_group_paths = [
                group_name for group_name in merged_dataset.group_paths
                if group_name in grid_image
            ]

        merged_group_paths = list(dict.fromkeys(merged_group_paths))
        merged_name_set = set(merged_group_paths)
        return {
            'train': merged_group_paths,
            'test': [name for name in all_image_names if name not in merged_name_set]
        }
        
class GridImage:
    def __init__(self, 
                 hdf5_path:Path|str,
                 group_name:str,
                 image_paths:List[str]|List[Path], 
                 mask_path:List[str]|List[Path],
                 meta_info:dict={},
                 image_name:str|None=None,
                 grid_size:Tuple|int=(2, 2),
                 grid_target_size:Tuple|int=(1024, 1024),
                 field_target_size:Tuple|int=(1536, 1536),
                 margin_ratio:float|list|tuple=0.1,
                 rescale:float=1.0,
                 generate_classification:dict={},
                 type_mode:Literal['inst_wise', 'signal_wise']='inst_wise',
                 cell_target_size:Tuple|int=(224, 224),
                 extension_rate:float=0.2,
                 object_attri_limit:dict|None=None, # {'attri_name': (lower_limit, upper_limit)}
                 defined_attri_func:list[callable]|None=None, # Described as extra properties for regionprops
                 normalize:bool=False,
                 delete_edges:bool=False,
                 input_z_size:int=10,
                 image_axis_order:str|list[str]='auto'): 
        

        self.hdf5_path = hdf5_path
        self.group_name = group_name
        self.image_name = group_name if image_name is None else image_name
        self.image_paths, self.mask_paths = self.check_file_paths(image_paths, mask_path)
        self.meta_info = dict(meta_info)
        self.grid_size = (grid_size, grid_size) if type(grid_size) is int else grid_size
        self.grid_target_size = (grid_target_size, grid_target_size) if type(grid_target_size) is int else grid_target_size
        self.field_target_size = (field_target_size, field_target_size) if type(field_target_size) is int else field_target_size
        self.margin_ratio = margin_ratio
        self.rescale = rescale
        self.generate_classification = generate_classification if generate_classification else {}
        self.type_mode = type_mode
        self.cell_target_size = (cell_target_size, cell_target_size) if isinstance(cell_target_size, int) else cell_target_size
        self.extension_rate = extension_rate
        self.object_attri_limit = object_attri_limit
        self.defined_attri_func = defined_attri_func
        self.normalize = normalize
        self.delete_edges = delete_edges
        self.input_z_size = input_z_size
        self.image_axis_order = image_axis_order
        self.input_dims = 3
        self.image_channel_metadata = {}
        
        self.enable_geo_aug = True
        self.enable_pix_aug = True

        self.stitch_info = {
            'patch_h': 0,
            'patch_w': 0,
            'margin_h': 0,
            'margin_w': 0,
            'original_h': 0,
            'original_w': 0
        }
        
        self.dealing_divide(
            generate_classification=self.generate_classification
        )

        self.grid_index = np.indices(self.grid_size).reshape(2, -1).T
        
        self.patch_dataset = GridDataSet(
            hdf5_path=self.hdf5_path,
            group_paths=[self.group_name],
            apply_augmentations=True,
            target_size=self.grid_target_size,
            type_num = 1 + len(self.generate_classification.get('channel', [])),
        )
        self.sc_dataset = scDataSet(
            hdf5_path=self.hdf5_path,
            group_paths=[self.group_name],
            apply_augmentations=True,
            target_size=self.cell_target_size
        )

    def __getitem__(self, key) -> dict[str, List[torch.Tensor|int]]:
        """
        Gets items from the grid dataset using 2D coordinates or slices.

        This method allows accessing patches using standard Python indexing and slicing
        on a 2D grid representation.

        Args:
            key: The index or slice. Can be one of:
                - An integer: `grid_image[i]` -> Returns the i-th patch linearly.
                - A tuple of two integers: `grid_image[row, col]` -> Returns the patch at the specified (row, col).
                - A tuple containing slices: `grid_image[0, :]` -> Returns all patches in the first row.
                                            `grid_image[:, 0:2]` -> Returns all patches in the first two columns.

        Returns:
            A dictionary representing a single patch, or a list of such dictionaries if a slice was used.
        """
        if isinstance(key, int):
            # Case 1: Linear indexing (e.g., grid_image[5])
            return self.patch_dataset.__getitem__(
                key, geo_aug=self.enable_geo_aug, pix_aug=self.enable_pix_aug)

        elif not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("GridImage indexing requires a single integer or a tuple of two indices")

        else:
            row_key, col_key = key
            num_rows, num_cols = self.grid_size
            assert isinstance(row_key, int) and isinstance(col_key, int), "GridImage indices must be integers"
            assert 0 <= row_key < num_rows and 0 <= col_key < num_cols, "GridImage index out of range"
            linear_index = row_key * num_cols + col_key
            return self.patch_dataset.__getitem__(
                linear_index, geo_aug=self.enable_geo_aug, pix_aug=self.enable_pix_aug)

    def __len__(self):
        """Returns the total number of patches in the grid."""
        return self.grid_index.shape[0]

    def dealing_divide(self, generate_classification:dict):
        """Divide the images into patches based on the grid size.

        Args:
            generate_classification (dict|None): If provided, will generate classification labels for each patch.
                If dict, in the format as: {'channel': [int], 'threshold': [float|Literal['auto']]}, where 'channel' is the channel index to use for classification,\
                and 'threshold' is the threshold value for classification. If 'auto', will use Otsu's method to determine the threshold.
        """
        total_img, total_mask, original_size, image_metadata = self.prepare_img_mask_dim(
            self.image_paths, 
            self.mask_paths[0],
            object_attri_limit=self.object_attri_limit,
            defined_attri_func=self.defined_attri_func,
            image_axis_order=self.image_axis_order,
            return_metadata=True)
        self.input_dims = image_metadata['input_dims']
        self.image_channel_metadata = image_metadata
        
        if self.delete_edges:
            mask_2d = np.squeeze(total_mask)
            footprint = np.ones((3, 3), dtype=bool)
            max_f = maximum_filter(mask_2d, footprint=footprint)
            min_f = minimum_filter(mask_2d, footprint=footprint)
            # Edges are where the local max is different from local min AND the pixel itself is not background
            edges = (mask_2d > 0) & (max_f != min_f)
            total_mask[0][edges] = 0
        
        # Now total_img is a list of channel volumes, each shaped as (Z, H, W).
        
        # --- Optimization: Generate type_array only ONCE on original resolution data ---
        type_array = self.generate_type(total_img, total_mask, 
                                        channel=generate_classification.get('channel', []), 
                                        threshold=generate_classification.get('threshold', []),
                                        type_mode=self.type_mode)

        cyto_type_class_weights, cyto_type_class_counts = self.compute_inst_wise_type_weights(total_mask, type_array)
        
        if self.normalize:
            total_img = self.normalize_image(total_img)

        # --- Extract single cell data BEFORE resizing for grid division, reusing the type_array ---
        sc_images, sc_coords, sc_cell_types = self.extract_single_cell_data(
            total_img, total_mask, type_array, self.extension_rate, generate_classification, self.cell_target_size
        )

        # Store original dimensions before potential resizing
        pre_resize_h, pre_resize_w = int(total_img[0].shape[-2]), int(total_img[0].shape[-1])

        total_img, total_mask, patch_h, patch_w, margin_h, margin_w, grid_size, margin_ratio = self.prepare_grid_divide(
            total_img, total_mask, self.margin_ratio, self.grid_size, self.field_target_size, rescale=self.rescale)

        if self.grid_size == 'auto':
            self.grid_size = grid_size
            self.margin_ratio = margin_ratio
        
        hv_array = generate_patch_hvmaps_2d(np.squeeze(total_mask))
        # If resizing occurred, resize the type_array as well
        post_resize_h, post_resize_w = int(total_img[0].shape[-2]), int(total_img[0].shape[-1])
        if (post_resize_h, post_resize_w) != (pre_resize_h, pre_resize_w):
            type_array_resized = np.zeros((type_array.shape[0], post_resize_h, post_resize_w), dtype=type_array.dtype)
            for i in range(type_array.shape[0]):
                type_array_resized[i] = cv2.resize(type_array[i], (post_resize_w, post_resize_h), interpolation=cv2.INTER_NEAREST)
            type_array = type_array_resized
        
        self.stitch_info['patch_h'] = patch_h
        self.stitch_info['patch_w'] = patch_w
        self.stitch_info['margin_h'] = margin_h
        self.stitch_info['margin_w'] = margin_w
        self.stitch_info['original_h'] = original_size[0]
        self.stitch_info['original_w'] = original_size[1]

        with h5py.File(self.hdf5_path, 'a') as f:
            if self.group_name in f:
                del f[self.group_name]
            group = f.create_group(self.group_name)

            if self.meta_info:
                group.attrs['meta_info'] = str(self.meta_info)
            group.attrs['grid_size'] = self.grid_size
            group.attrs['cyto_type_class_weights'] = cyto_type_class_weights.astype(np.float32)
            group.attrs['cyto_type_class_counts'] = cyto_type_class_counts.astype(np.int64)
            group.attrs['input_dims'] = 3
            group.attrs['input_z_size'] = self.input_z_size
            group.attrs['image_channel_count'] = image_metadata['channel_count']
            group.attrs['channel_shapes'] = str(image_metadata['channel_shapes'])
            group.attrs['channel_axis_orders'] = str(image_metadata['channel_axis_orders'])
            group.attrs['channel_z_sizes'] = str(image_metadata['channel_z_sizes'])
            group.attrs['channel_source_paths'] = str(image_metadata['channel_source_paths'])

            # --- Save single-cell stack ---
            if sc_images is not None:
                sc_group = group.create_group('sc_stack')
                sc_group.create_dataset('coords', data=sc_coords, compression="gzip")
                sc_group.create_dataset('cell_types', data=sc_cell_types, compression="gzip")
                sc_group.attrs['meta_info'] = str(self.meta_info)
                sc_group.attrs['input_dims'] = 3
                sc_group.attrs['input_z_size'] = self.input_z_size
                sc_group.attrs['channel_z_sizes'] = str(image_metadata['channel_z_sizes'])
                cells_group = sc_group.create_group('cells')
                for cell_idx, cell_channels in enumerate(sc_images):
                    cell_group = cells_group.create_group(f"cell_{cell_idx}")
                    raw_group = cell_group.create_group('raw_channels')
                    for ch_idx, channel_image in enumerate(cell_channels):
                        ch_group = raw_group.create_group(f"channel_{ch_idx}")
                        ch_group.create_dataset('image', data=channel_image, compression="gzip")

            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    patch_group = group.create_group(f"patch_{i}_{j}")

                    row_slice = slice((i * (patch_h - margin_h)), ((i + 1) * patch_h - i * margin_h))
                    col_slice = slice((j * (patch_w - margin_w)), ((j + 1) * patch_w - j * margin_w))
                    patch_img = [channel[:, row_slice, col_slice] for channel in total_img]
                    patch_mask = total_mask[:, (i * (patch_h - margin_h)):((i + 1) * patch_h - i * margin_h),
                                            (j * (patch_w - margin_w)):((j + 1) * patch_w - j * margin_w)]
                    patch_hv = hv_array[:, (i * (patch_h - margin_h)):((i + 1) * patch_h - i * margin_h),
                                        (j * (patch_w - margin_w)):((j + 1) * patch_w - j * margin_w)]
                    
                    patch_group.attrs['input_z_size'] = self.input_z_size
                    patch_group.attrs['channel_z_sizes'] = str(image_metadata['channel_z_sizes'])
                    raw_group = patch_group.create_group('raw_channels')
                    for ch_idx, channel_image in enumerate(patch_img):
                        ch_group = raw_group.create_group(f"channel_{ch_idx}")
                        ch_group.create_dataset('image', data=channel_image, compression="gzip")
                    patch_group.create_dataset('mask', data=patch_mask, compression="gzip")
                    patch_group.create_dataset('hv_map', data=patch_hv, compression="gzip")

                    patch_type = type_array[:, (i * (patch_h - margin_h)):((i + 1) * patch_h - i * margin_h),
                                            (j * (patch_w - margin_w)):((j + 1) * patch_w - j * margin_w)]
                    patch_group.create_dataset('type_map', data=patch_type, compression="gzip")
        
        return

    def check_file_paths(self, 
                         image_paths:List[str]|List[Path]|str|Path, 
                         mask_paths:List[str]|List[Path]|str|Path,
                         ) -> Tuple[List[Path], List[Path]]:
        """Check if the provided file paths exist."""

        if type(image_paths) in [str, Path]:
            if not os.path.exists(image_paths):
                raise ValueError(f"Image path {image_paths} does not exist.")
            elif os.path.isdir(image_paths):
                image_paths = [Path(os.path.join(image_paths, f)) for f in os.listdir(image_paths) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
            else:
                image_paths = [Path(image_paths)]

        elif isinstance(image_paths, list):
            image_paths_ = [Path(p) for p in image_paths if os.path.exists(p)]
            if len(image_paths_) < len(image_paths):
                warnings.warn("Some image paths do not exist. Only existing paths will be used.")
            image_paths = image_paths_
        else:
            raise ValueError("image_paths must be a string, Path, or a list of strings/Paths.")
        
        assert len(image_paths) > 0, "No valid image paths provided. Please check the input paths."
        
        if type(mask_paths) in [str, Path, None]:
            if (not os.path.exists(mask_paths)) or (mask_paths is None):
                warnings.warn(f"Mask path {mask_paths} does not exist. No masks will be used.")
                mask_paths = []
            elif os.path.isdir(mask_paths):
                mask_paths = [Path(os.path.join(mask_paths, f)) for f in os.listdir(mask_paths) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
            else:
                mask_paths = [Path(mask_paths)]

        elif isinstance(mask_paths, list):
            mask_paths_ = [Path(p) for p in mask_paths if os.path.exists(p)]
            if len(mask_paths_) < len(mask_paths):
                warnings.warn("Some mask paths do not exist. Only existing paths will be used.")
            mask_paths = mask_paths_

        else:
            raise ValueError("mask_paths must be a string, Path, or a list of strings/Paths.")
        
        return image_paths, mask_paths

    def compute_inst_wise_type_weights(self, mask: np.ndarray, type_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-class inverse instance-frequency weights for one GridImage."""

        if self.type_mode == 'inst_wise':
            instance_mask = np.squeeze(mask).astype(np.int64)
            num_classes = type_array.shape[0]
            inst_labels = np.unique(instance_mask)
            inst_labels = inst_labels[inst_labels > 0]

            if inst_labels.size == 0:
                class_counts = np.zeros(num_classes, dtype=np.int64)
            else:
                type_sums = np.zeros((inst_labels.size, num_classes), dtype=np.float64)
                for class_idx in range(num_classes):
                    type_sums[:, class_idx] = np.asarray(
                        sum_labels(type_array[class_idx], labels=instance_mask, index=inst_labels),
                        dtype=np.float64)
                instance_classes = np.argmax(type_sums, axis=1)
                instance_classes[type_sums.sum(axis=1) <= 0] = 0
                class_counts = np.bincount(instance_classes, minlength=num_classes)[:num_classes].astype(np.int64)

            class_freq = class_counts.astype(np.float64) / max(class_counts.sum(), 1)
            positive_freq = class_freq[class_freq > 0]
            mean_positive_freq = positive_freq.mean() if positive_freq.size > 0 else 1.0 / num_classes
            safe_freq = np.where(class_freq > 0, class_freq, mean_positive_freq)
            raw_weights = mean_positive_freq / safe_freq
            clipped_weights = np.clip(raw_weights, 0.2, 8.0)
            class_weights = clipped_weights / clipped_weights.mean()
            return class_weights.astype(np.float32), class_counts
        else:
            return np.ones(type_array.shape[0], dtype=np.float32), np.zeros(type_array.shape[0], dtype=np.int64)

    @staticmethod
    def prepare_img_mask_dim(image_path:str|Path, 
                             mask_path:str|Path,
                             object_attri_limit:dict[str, tuple]|None=None,
                             defined_attri_func:list[Callable]|None=None,
                             image_axis_order:str|list[str]='auto',
                             return_metadata:bool=False
                             )-> Tuple[list[np.ndarray], np.ndarray, Tuple[int, int]]:
        """Prepare image channels, optional z metadata, and a 2D instance mask for processing."""

        image, image_metadata = GridImage.read_image_channels(image_path, image_axis_order=image_axis_order)
        mask:np.ndarray = np.squeeze(imread(mask_path))
        original_size = mask.shape
        if len(original_size) > 2:
            raise ValueError("Mask images should be single-channel (grayscale)")

        if defined_attri_func is not None:
            if type(defined_attri_func) is not list:
                defined_attri_func = [defined_attri_func]

        if object_attri_limit is not None:
            obj_props:list[RegionProperties] = regionprops(mask, extra_properties=defined_attri_func)
            label2erase = []
            for key in object_attri_limit.keys():
                lower_limit, upper_limit = object_attri_limit[key]
                if upper_limit is not None or lower_limit is not None:
                    for prop in obj_props:
                        prop_val = getattr(prop, key)
                        if upper_limit is not None and prop_val > upper_limit:
                            label2erase.append(prop.label)
                        if lower_limit is not None and prop_val < lower_limit:
                            label2erase.append(prop.label)
                if len(label2erase) > 0:
                    mask = np.where(np.isin(mask, label2erase), 0, mask)
            mask = relabel_sequential(mask)[0]

        mask = np.expand_dims(mask, axis=0)

        image_h, image_w = int(image[0].shape[-2]), int(image[0].shape[-1])
        assert image_h == mask.shape[1] and image_w == mask.shape[2], \
            f"Image and mask dimensions do not match. Image shape: {(image_h, image_w)}, Mask shape: {mask.shape}"

        if return_metadata:
            return image, mask, original_size, image_metadata
        return image, mask, original_size

    @staticmethod
    def read_image_channels(image_paths: str|Path|list[str]|list[Path],
                            image_axis_order: str|list[str] = 'auto') -> tuple[list[np.ndarray], dict]:
        """Read image paths into raw channel volumes; every channel is shaped as (Z, H, W)."""
        image_paths = [image_paths] if type(image_paths) in [str, Path] else list(image_paths)
        if isinstance(image_axis_order, list):
            axis_orders = image_axis_order
        else:
            axis_orders = [image_axis_order] * len(image_paths)

        channel_arrays = []
        channel_metadata = []
        split_single_file_channels = len(image_paths) == 1
        for path, axis_order in zip(image_paths, axis_orders):
            image = imread(path)
            channels, metadata = GridImage._standardize_image_array(
                image,
                source_path=str(path),
                image_axis_order=axis_order,
                split_channel_axis=split_single_file_channels)
            channel_arrays.extend(channels)
            channel_metadata.extend(metadata)

        metadata_out = {
            'input_dims': 3,
            'source_input_dims': 3 if any(channel.shape[0] > 1 for channel in channel_arrays) else 2,
            'channel_count': len(channel_arrays),
            'channel_shapes': [meta['raw_shape'] for meta in channel_metadata],
            'channel_axis_orders': [meta['axis_order'] for meta in channel_metadata],
            'channel_z_sizes': [meta['z_size'] for meta in channel_metadata],
            'channel_source_paths': [meta['source_path'] for meta in channel_metadata],
        }
        return channel_arrays, metadata_out

    @staticmethod
    def _standardize_image_array(image: np.ndarray,
                                 source_path: str,
                                 image_axis_order: str = 'auto',
                                 split_channel_axis: bool = True) -> tuple[list[np.ndarray], list[dict]]:
        """Convert one loaded image array into one or more channel volumes shaped as (Z, H, W)."""
        axis_order = image_axis_order.lower() if isinstance(image_axis_order, str) else 'auto'
        channels = []
        metadata = []

        if image.ndim == 2:
            channel_arrays = [np.expand_dims(image, axis=0)]
            resolved_axis = ['YX']
        elif image.ndim == 3 and axis_order == 'auto' and split_channel_axis and image.shape[-1] in (1, 3, 4):
            image_cyx = np.moveaxis(image, -1, 0)
            channel_arrays = [np.expand_dims(image_cyx[ch_idx], axis=0) for ch_idx in range(image_cyx.shape[0])]
            resolved_axis = ['YXC'] * len(channel_arrays)
        elif image.ndim == 3 and axis_order in ('zyx', 'auto'):
            channel_arrays = [image]
            resolved_axis = ['ZYX']
        elif image.ndim == 3 and axis_order == 'yxc':
            image_cyx = np.moveaxis(image, -1, 0)
            channel_arrays = [np.expand_dims(image_cyx[ch_idx], axis=0) for ch_idx in range(image_cyx.shape[0])]
            resolved_axis = ['YXC'] * len(channel_arrays)
        elif image.ndim == 3 and axis_order == 'cyx':
            channel_arrays = [np.expand_dims(image[ch_idx], axis=0) for ch_idx in range(image.shape[0])]
            resolved_axis = ['CYX'] * len(channel_arrays)
        elif image.ndim == 4 and axis_order in ('czyx', 'auto'):
            channel_arrays = [image[ch_idx] for ch_idx in range(image.shape[0])]
            resolved_axis = ['CZYX'] * len(channel_arrays)
        else:
            raise ValueError(f"Unsupported image shape {image.shape} with axis order {image_axis_order}.")

        for channel_array, resolved in zip(channel_arrays, resolved_axis):
            channel_array = np.asarray(channel_array)
            if channel_array.ndim == 2:
                channel_array = np.expand_dims(channel_array, axis=0)
            channels.append(channel_array)
            metadata.append({
                'source_path': source_path,
                'raw_shape': tuple(channel_array.shape),
                'axis_order': resolved,
                'z_size': int(channel_array.shape[0])
            })
        return channels, metadata

    @staticmethod
    def generate_type(channels: list[np.ndarray], mask, channel:list, threshold:list, type_mode:Literal['inst_wise', 'signal_wise']='inst_wise') -> np.ndarray:
        """
        Generate a type mask including a baseline type.

        Args:
            channels (list[np.ndarray]): Channel volumes, each shaped (Z, H, W).
            mask (np.ndarray): The instance mask array.
            channel (list): List of channel indices for classification.
            threshold (list): List of threshold values for classification.

        Returns:
            np.ndarray: A combined type mask array of shape ((L + 1), H, W).
        """
        instance_mask_int = np.squeeze(mask).astype(np.uint16)
        boolean_mask = instance_mask_int > 0
        image = np.moveaxis(
            np.stack([np.max(channel, axis=0) for channel in channels], axis=0), 0, -1)  # (H, W, C)
        num_specific_types = len(channel)
        num_total_types = num_specific_types + 1

        # Total shape: 2 * (L + 1)
        type_array = np.zeros((num_total_types, boolean_mask.shape[0], boolean_mask.shape[1]), dtype=np.float32)

        

        if num_specific_types == 0:
            return type_array

        # --- Logic for specific types ---
        if not isinstance(channel, list): channel = [channel]
        if not isinstance(threshold, list): threshold = [threshold]
        if len(channel) > 1 and len(threshold) == 1:
            threshold = threshold * len(channel)
        assert len(channel) == len(threshold), "channel and threshold must have the same length."

        if type_mode == 'inst_wise':
            type_array[0, boolean_mask] = 1.0  # cyto baseline (channel 0)
            max_label = instance_mask_int.max()
            if max_label == 0:
                return type_array
                
            labels_range = np.arange(1, max_label + 1)
            
            # intensity_mat shape: (max_label, num_specific_types)
            intensity_mat = np.zeros((max_label, num_specific_types), dtype=np.float32)
            
            counts = sum_labels(np.ones_like(instance_mask_int), labels=instance_mask_int, index=labels_range)
            existing_mask = counts > 0

            for i, ch_idx in enumerate(channel):
                # Calculate sum of intensities for each label
                sums = sum_labels(image[:, :, ch_idx], labels=instance_mask_int, index=labels_range)
                
                # Calculate means, handling division by zero for missing labels
                with np.errstate(divide='ignore', invalid='ignore'):
                    means = sums / counts
                    means[~existing_mask] = 0 # Set mean to 0 for non-existing labels
                
                intensity_mat[:, i] = means

            channel_thres = np.zeros(num_specific_types, dtype=np.float32)
            for i, th in enumerate(threshold):
                if th == 'auto':
                    valid_intensities = intensity_mat[existing_mask, i]
                    if valid_intensities.size > 0:
                        channel_thres[i] = threshold_otsu(valid_intensities)
                    else:
                        channel_thres[i] = 0
                else:
                    channel_thres[i] = th

            for c in range(num_specific_types):
                
                current_thres = channel_thres[c]
                
                if current_thres > 0:
                    positive_indices = intensity_mat[:, c] > current_thres
                else:
                    positive_indices = intensity_mat[:, c] < abs(current_thres)

                # Optimization: Use LUT (Look-Up Table) for fast mask generation

                lut = np.zeros(max_label + 1, dtype=bool)
                lut[1:] = positive_indices
                
                # Apply LUT to create boolean mask
                positive_mask = lut[instance_mask_int]

                if np.any(positive_mask):
                    type_array[1 + c, positive_mask] = 1.0
                    type_array[0, positive_mask] = 0.0
        
        elif type_mode == 'signal_wise':
            type_array[0, :] = 1.0
            for c, (ch_idx, th) in enumerate(zip(channel, threshold)):
                img_ch = image[:, :, ch_idx]
                
                if th == 'auto':
                    # Use pixels within instances to determine threshold
                    valid_pixels = img_ch[boolean_mask]
                    if valid_pixels.size > 0:
                        current_thres = threshold_otsu(img_ch)
                    else:
                        current_thres = 0
                else:
                    current_thres = th
                
                if current_thres > 0:
                    positive_mask = img_ch >= current_thres
                else:
                    positive_mask = img_ch < abs(current_thres)
                
                if np.any(positive_mask):
                    type_array[1 + c, positive_mask] = 1.0
                    type_array[0, positive_mask] = 0.0
        
        return type_array

    @staticmethod
    def normalize_image(channels: list[np.ndarray]) -> list[np.ndarray]:
        """Z-score normalize each channel volume (Z, H, W), clip to [-3, 3], then scale to [0, 1]."""
        normalized = []
        for channel in channels:
            ch = channel.astype(np.float32)
            mean = np.mean(ch)
            std = np.std(ch)
            if std > 1e-6:
                ch = (ch - mean) / std
                ch = np.clip(ch, -3, 3)
                ch = (ch + 3) / 6
            else:
                ch = np.zeros_like(ch)
            normalized.append(ch)
        return normalized

    @staticmethod
    def extract_single_cell_data(channels: list[np.ndarray], mask: np.ndarray, type_array: np.ndarray, extension_rate: float, generate_classification: dict, cell_target_size: Tuple[int, int]):
        """Extracts image patches for each single cell and resizes them to a target size."""
        instance_mask = np.squeeze(mask).astype(np.uint16)
        props = regionprops(instance_mask)
        if not props:
            return None, None, None

        sc_images = []
        
        # --- Vectorized Extraction of Centroids and Type Distributions ---
        inst_labels = np.array([p.label for p in props])
        sc_coords = np.array([p.centroid for p in props])
        sc_cell_types = GridImage.get_type_distribution_vectorized(instance_mask, type_array, inst_labels)
        # --- End of Vectorized Part ---

        for prop in tqdm(props, desc="Cropping & Resizing Cells"):
            min_row, min_col, max_row, max_col = prop.bbox
            
            height, width = max_row - min_row, max_col - min_col
            ext_h, ext_w = int(height * extension_rate / 2), int(width * extension_rate / 2)

            # Padded crop
            pad_min_r, pad_min_c = max(0, min_row - ext_h), max(0, min_col - ext_w)
            image_h, image_w = int(channels[0].shape[-2]), int(channels[0].shape[-1])
            pad_max_r, pad_max_c = min(image_h, max_row + ext_h), min(image_w, max_col + ext_w)
            row_slice = slice(pad_min_r, pad_max_r)
            col_slice = slice(pad_min_c, pad_max_c)
            cropped_channels = [channel[:, row_slice, col_slice] for channel in channels]

            # Handle padding if crop is at the edge
            pad_top = max(0, ext_h - min_row)
            pad_bottom = max(0, (max_row + ext_h) - image_h)
            pad_left = max(0, ext_w - min_col)
            pad_right = max(0, (max_col + ext_w) - image_w)

            if any(p > 0 for p in [pad_top, pad_bottom, pad_left, pad_right]):
                cropped_channels = [
                    np.pad(channel, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), 'constant')
                    for channel in cropped_channels
                ]

            sc_images.append([
                GridImage._resize_channel_volume_xy(channel, cell_target_size)
                for channel in cropped_channels
            ])

        if not sc_images:
            return None, None, None

        return sc_images, sc_coords, sc_cell_types

    @staticmethod
    def get_type_distribution_vectorized(inst_map: np.ndarray, type_map: np.ndarray, inst_labels: np.ndarray) -> np.ndarray:
        """
        Calculates the type distribution for each instance in a vectorized manner.
        
        Args:
            inst_map (np.ndarray): The instance segmentation map (H, W).
            type_map (np.ndarray): The type map (C, H, W), where C is the number of types.
            inst_labels (np.ndarray): A 1D array of unique instance labels to calculate distributions for.

        Returns:
            np.ndarray: A (N, C) array where N is the number of instances and C is the number of types,
                        containing the proportional distribution of each type within each instance.
        """
        from scipy import ndimage

        num_types = type_map.shape[0]
        
        # Get total area (pixel count) for each instance label
        # We need to ensure inst_labels is sorted for indexing `ndimage.sum_labels`
        sorted_inst_labels = np.sort(inst_labels)
        instance_areas = ndimage.sum_labels(np.ones_like(inst_map), labels=inst_map, index=sorted_inst_labels)
        
        # Create a mapping from original label order to sorted order for later re-ordering
        sort_order = np.argsort(inst_labels)
        unsort_order = np.argsort(sort_order)

        # Calculate sum of pixels for each type within each instance
        type_counts_per_instance = np.zeros((len(inst_labels), num_types))

        for i in range(num_types):
            # For each type channel, sum the values within each instance region
            type_pixel_counts = ndimage.sum_labels(type_map[i], labels=inst_map, index=sorted_inst_labels)
            type_counts_per_instance[:, i] = type_pixel_counts

        # Avoid division by zero for instances with zero area
        instance_areas[instance_areas == 0] = 1
        
        # Calculate the distribution
        distributions = type_counts_per_instance / instance_areas[:, np.newaxis]
        
        # Return the distributions in the original order of inst_labels
        return distributions[unsort_order]

    @staticmethod
    def _resize_channel_volume_xy(channel: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
        """Resize each z slice in one channel volume to the requested XY shape."""
        target_h, target_w = target_hw
        resized = np.zeros((channel.shape[0], target_h, target_w), dtype=channel.dtype)
        for z_idx in range(channel.shape[0]):
            resized[z_idx] = cv2.resize(channel[z_idx], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return resized

    @staticmethod
    def prepare_grid_divide(channels: list[np.ndarray], mask:np.ndarray, margin_ratio:float|tuple|list, 
                            grid_size:Tuple[int, int]|str, field_target_size:Tuple[int, int], rescale:float=1.0):
        """ Define the grid parameters based on the image and mask dimensions.

        Args:
            channels (list[np.ndarray]): Channel volumes, each shaped (Z, H, W).
            mask (np.ndarray): The instance mask array.
            margin_ratio (float|tuple|list): The ratio of the margin to be applied to each patch.
            grid_size (Tuple[int, int]): The number of patches in the grid (rows, columns).
            rescale (float): Rescale factor for the image and mask.

        Returns:
            Tuple(np.ndarray, np.ndarray): The resized image and mask arrays.

        """
        image_h, image_w = int(channels[0].shape[-2]), int(channels[0].shape[-1])
        if rescale != 1.0:
            h, w  = int(image_h * rescale), int(image_w * rescale)
        else:
            h, w = image_h, image_w

        if grid_size == 'auto':
            real_field_target_size = [0, 0]
            if field_target_size is None:
                raise ValueError("field_target_size must be provided when grid_size is 'auto'.")
            else:
                real_field_target_size[0] = field_target_size[0]
                real_field_target_size[1] = field_target_size[1]
            if real_field_target_size[0] > h:
                real_field_target_size[0] = h
            if real_field_target_size[1] > w:
                real_field_target_size[1] = w
            init_guess_grid = \
                [int(np.ceil(h / real_field_target_size[0])), int(np.ceil(w / real_field_target_size[1]))]
            init_guess_mr = \
                [0, 0]
            
            set_h = False if init_guess_grid[0] > 1 else True
            set_w = False if init_guess_grid[1] > 1 else True
            while not (set_h and set_w):
                if not set_h:
                    guess_mr_h = (init_guess_grid[0] - h / real_field_target_size[0]) / (init_guess_grid[0] - 1)
                    if guess_mr_h < 0.1:
                        init_guess_grid[0] += 1
                    else:
                        init_guess_mr[0] = guess_mr_h
                        set_h = True
                else:
                    pass
                if not set_w:
                    guess_mr_w = (init_guess_grid[1] - w / real_field_target_size[1]) / (init_guess_grid[1] - 1)
                    if guess_mr_w < 0.1:
                        init_guess_grid[1] += 1
                    else:
                        init_guess_mr[1] = guess_mr_w
                        set_w = True
                else:
                    pass
            grid_size = (init_guess_grid[0], init_guess_grid[1])
            margin_ratio = (init_guess_mr[0], init_guess_mr[1])

        if isinstance(margin_ratio, float):
            margin_ratio = (margin_ratio, margin_ratio)

        patch_h = int(h / ( (1 - margin_ratio[0]) * grid_size[0] + margin_ratio[0]))
        patch_w = int(w / ( (1 - margin_ratio[1]) * grid_size[1] + margin_ratio[1]))
        margin_h = int(patch_h * margin_ratio[0])
        margin_w = int(patch_w * margin_ratio[1])

        actual_total_h = (patch_h - margin_h) * grid_size[0] + margin_h
        actual_total_w = (patch_w - margin_w) * grid_size[1] + margin_w

        if actual_total_h != image_h or actual_total_w != image_w:
            target_hw = (actual_total_h, actual_total_w)
            channels = [GridImage._resize_channel_volume_xy(channel, target_hw) for channel in channels]
            mask = cv2.resize(np.squeeze(mask),
                               (actual_total_w, actual_total_h), interpolation=cv2.INTER_NEAREST)

            return channels, np.expand_dims(mask, axis=0), patch_h, patch_w, margin_h, margin_w, grid_size, margin_ratio
        
        else:
            return channels, mask, patch_h, patch_w, margin_h, margin_w, grid_size, margin_ratio
    
class GridDataSet(Dataset):
    def __init__(self, 
                 hdf5_path: str | Path,
                 group_paths: List[str],
                 apply_augmentations:bool=True,
                 target_size:Tuple[int, int]=(512, 512),
                 type_num:int=1):
        super().__init__()
        self.hdf5_path = Path(hdf5_path)
        self.group_paths = group_paths
        self.apply_augmentations = apply_augmentations
        self.target_size = target_size
        self.type_num = type_num
        self.input_dims = 3
        self.input_z_size = 10
        self.image_channel_count = 0
        self.target_image_channel = None
        
        self.patch_info = [] # This will be populated by _get_patch_info
        self._get_patch_info()

        if self.apply_augmentations:
            # Geometric augmentation magnitudes (the original settings suffice for the label-free specialist model).
            self.rotation_degrees = (-90, 90)
            self.scale_factor_range = (0.75, 1.50)               # TF.affine scaling
            # Photometric augmentation magnitudes
            self.brightness_factor_range = (0.5, 1.25)
            self.contrast_factor_range = (0.8, 1.3)
            # Blur: allow a larger sigma; adapt the kernel to sigma at runtime (see apply_transforms).
            self.blur_sigma_range = (0.5, 2.0)
            # Downsampling-upsampling: simulate undersampling / binning.
            self.downscale_factor_range = (0.5, 0.95)
            # Uneven illumination: relative amplitude and control-point grid size of a low-frequency multiplicative field.
            self.illumination_strength_range = (0.15, 0.45)
            self.illumination_grid = 4
            # Additive noise: smaller Poisson scale means stronger noise; Gaussian uses a relative standard deviation.
            self.poisson_noise_scale_range = (60.0, 200.0)
            self.gaussian_noise_std_range = (0.01, 0.06)
            # Augmentation probabilities, centralized for easier tuning.
            self.aug_prob = {
                'rotate': 0.5, 'hflip': 0.5, 'vflip': 0.5, 'scale': 0.5,
                'invert': 0.0, 'brightness': 0.5, 'contrast': 0.5,
                'blur': 0.5, 'downscale': 0.5, 'illumination': 0.5, 'noise': 0.5,
            }
            # Training-time percentile normalization (Cellpose style).
            # Keep False if upstream GridImage already applies whole-image z-score (normalize=True),
            # to preserve relative patch intensities and match inference using only whole-image z-score.
            self.normalize_percentile = False
            self.norm_percentiles = (1.0, 99.0)

    def configure_3d_input(self, target_image_channel:list[int]|int|None=None, input_z_size:int=10) -> None:
        """Set the biological channels used for lazy 3D z-size checks and z resampling."""
        if isinstance(target_image_channel, int):
            target_image_channel = [target_image_channel]
        self.target_image_channel = target_image_channel
        self.input_z_size = input_z_size

    def _get_patch_info(self):
        """Retrieve patch paths and metadata from HDF5 file."""
        self.patch_info = []
        if not self.hdf5_path.exists():
            warnings.warn(f"HDF5 file not found at {self.hdf5_path}, cannot load patch info.")
            return
            
        with h5py.File(self.hdf5_path, 'r') as f:
            for group_path in self.group_paths:
                if group_path not in f:
                    warnings.warn(f"Group '{group_path}' not found in HDF5 file.")
                    continue
                group = f[group_path]
                self.input_dims = 3
                self.image_channel_count = max(self.image_channel_count, int(group.attrs.get('image_channel_count', 0)))
                meta_info = group.attrs.get('meta_info', '')
                meta_info = literal_eval(meta_info) if meta_info else {}
                cyto_type_class_weights = np.asarray(
                    group.attrs.get('cyto_type_class_weights', np.ones(self.type_num, dtype=np.float32)),
                    dtype=np.float32)
                if cyto_type_class_weights.shape[0] != self.type_num:
                    cyto_type_class_weights = np.ones(self.type_num, dtype=np.float32)

                if 'grid_size' in group.attrs:
                    grid_size = tuple(group.attrs['grid_size'])
                else:
                    # Infer grid_size from patch names if not in attrs
                    patch_keys = [k for k in group.keys() if k.startswith('patch_')]
                    max_i, max_j = -1, -1
                    for key in patch_keys:
                        try:
                            _, i, j = key.split('_')
                            max_i = max(max_i, int(i))
                            max_j = max(max_j, int(j))
                        except ValueError:
                            continue
                    grid_size = (max_i + 1, max_j + 1)

                for i in range(grid_size[0]):
                    for j in range(grid_size[1]):
                        patch_name = f"patch_{i}_{j}"
                        if patch_name in group:
                            self.patch_info.append({
                                'h5_path': f"{group_path}/{patch_name}",
                                'row': i,
                                'col': j,
                                'meta_info': meta_info,
                                'cyto_type_class_weights': cyto_type_class_weights,
                                'channel_z_sizes': literal_eval(group[patch_name].attrs.get('channel_z_sizes', '[]'))
                            })

    def __len__(self) -> int:
        """Returns the total number of images in the dataset."""
        return len(self.patch_info)

    def __getitem__(self, index:int, return_blank:bool=False, geo_aug:bool=True, pix_aug:bool=True) -> List[torch.Tensor|None]:
        if return_blank:
            channel_count = len(self.target_image_channel) if self.target_image_channel else 1
            image = np.zeros((channel_count, self.input_z_size, *self.target_size), dtype=np.float32)
            mask = np.zeros((1, *self.target_size), dtype=np.uint8)
            hv_map = np.zeros((4, *self.target_size), dtype=np.float32)
            type_map = np.zeros((self.type_num, *self.target_size), dtype=np.float32)
            total_img_list = [image, mask, hv_map, type_map]
            total_tensor_list = self.apply_transforms(total_img_list, geo_aug=False, pix_aug=False)
            image_channel_indices = list(range(image.shape[0]))
            info = {
                'row': -1,
                'col': -1,
                'meta_info': {},
                'cyto_type_class_weights': np.ones(self.type_num, dtype=np.float32)
            }
        else:
            info = self.patch_info[index]
            with h5py.File(self.hdf5_path, 'r') as f:
                patch_group = f[info['h5_path']]
                image, image_channel_indices = self._load_3d_patch_image(patch_group)
                mask = patch_group['mask'][:]
                hv_map = patch_group['hv_map'][:]
                type_map = patch_group['type_map'][:]
            if np.isnan(image).any() or np.isnan(mask).any():
                raise ValueError("NaN detected in loaded data")
            
            total_img_list = [image, mask, hv_map, type_map]
            total_tensor_list = self.apply_transforms(total_img_list, geo_aug=geo_aug, pix_aug=pix_aug)

        patch_meta_info = dict(info.get('meta_info', {}))
        patch_meta_info['patch_row'] = info['row']
        patch_meta_info['patch_col'] = info['col']
        cyto_type_class_weights = np.asarray(
            info.get('cyto_type_class_weights', np.ones(self.type_num, dtype=np.float32)),
            dtype=np.float32)

        return {'row': info['row'],
            'col': info['col'],
            'meta_info': patch_meta_info,
                'image': total_tensor_list[0],
                'instance': total_tensor_list[1],
                'mask': total_tensor_list[2][0:2, :, :].clone(),
                'hv_map': total_tensor_list[2][2:4, :, :].clone(),
                'type_cyto_map': total_tensor_list[3].clone(),
                'cyto_type_class_weights': torch.from_numpy(cyto_type_class_weights).float(),
                'image_channel_indices': torch.tensor(image_channel_indices, dtype=torch.long),
                'data_index': index,
                }

    def __add__(self, other: 'GridDataSet') -> 'GridDataSet':
        """Merge two GridDataSet instances."""
        if not isinstance(other, GridDataSet):
            raise TypeError("Can only merge with another GridDataSet instance.")
        
        return self.merge_dataset([self, other])

    @staticmethod
    def merge_dataset(dataset_list:List['GridDataSet']) -> 'GridDataSet':
        
        """Get a merged dataset from multiple grid images."""
        if not dataset_list:
            return None
            
        first_dataset = dataset_list[0]
        hdf5_path = first_dataset.hdf5_path
        target_size = first_dataset.target_size
        type_num = first_dataset.type_num

        all_group_paths = []
        for ds in dataset_list:
            if ds.hdf5_path != hdf5_path:
                raise ValueError("All datasets to be merged must share the same HDF5 file.")
            all_group_paths.extend(ds.group_paths)

        merged_dataset = GridDataSet(
            hdf5_path=hdf5_path,
            group_paths=all_group_paths, # Use set to remove duplicates
            apply_augmentations=True,
            target_size=target_size,
            type_num=type_num
        )
        return merged_dataset

    def _load_3d_patch_image(self, patch_group) -> tuple[np.ndarray, list[int]]:
        """Load selected raw channel volumes and z-resample them to (C, input_z_size, H, W)."""
        
        if self.target_image_channel is None:
            target_channels = [0]
        elif isinstance(self.target_image_channel, int):
            target_channels = [self.target_image_channel]
        else:
            target_channels = list(self.target_image_channel)
        
        channel_arrays = []
        z_sizes = []
        source_dtype = None
        for channel_idx in target_channels:
            channel_image = patch_group['raw_channels'][f'channel_{channel_idx}']['image'][:]
            source_dtype = channel_image.dtype if source_dtype is None else source_dtype
            channel_arrays.append(channel_image)
            z_sizes.append(int(channel_image.shape[0]))
        if len(set(z_sizes)) != 1:
            raise ValueError(
                f"Selected target_image_channel {target_channels} have inconsistent z sizes {z_sizes}."
            )

        z_raw = z_sizes[0]
        image_tensor = torch.from_numpy(np.stack(channel_arrays, axis=0)).float().unsqueeze(0)
        if z_raw != self.input_z_size:
            image_tensor = F.interpolate(
                image_tensor,
                size=(self.input_z_size, image_tensor.shape[-2], image_tensor.shape[-1]),
                mode='trilinear',
                align_corners=False)
        image_np = image_tensor.squeeze(0).numpy()
        if source_dtype is not None and np.issubdtype(source_dtype, np.integer):
            image_np = image_np / (float(np.iinfo(source_dtype).max) + 1e-6)
        return image_np.astype(np.float32), target_channels

    def apply_transforms(self, image_list:List[np.ndarray], geo_aug:bool=True, pix_aug:bool=True) -> List[torch.Tensor|None]:
        """Resize and augment patch tensors; treat planar (C, H, W) input as a volume with Z=1."""

        # ---- Load images as float: scale integers by the dtype maximum to approximately [0,1]. ----
        img_np = image_list[0]
        planar_input = img_np.ndim == 3
        if np.issubdtype(img_np.dtype, np.integer):
            denom = float(np.iinfo(img_np.dtype).max)
        else:
            denom = 1.0
        img = torch.from_numpy(img_np.astype(np.float32) / (denom + 1e-6)).float()
        if planar_input:
            img = img.unsqueeze(1)

        # ---- Label tensors: use large-value encoding for masks and cast hv/type directly to float. ----
        mask = torch.from_numpy(image_list[1] / 100000).float()
        hv_map = torch.from_numpy(image_list[2]).float()
        type_map = torch.from_numpy(image_list[3]).float() if image_list[3] is not None else None

        # ---- Resize to the target size: trilinear for images, label-specific interpolation for labels. ----
        img = self._resize_input_image(img, self.target_size)
        mask = TF.resize(mask, self.target_size, interpolation=InterpolationMode.NEAREST, antialias=False)
        hv_map = TF.resize(hv_map, self.target_size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        type_map = TF.resize(type_map, self.target_size, interpolation=InterpolationMode.NEAREST, antialias=True)

        # ---- Percentile normalization before degradation (Cellpose style): normalize low-contrast label-free images. ----
        if self.apply_augmentations and self.normalize_percentile:
            img = self._percentile_normalize(img, *self.norm_percentiles)

        if self.apply_augmentations:
            if geo_aug:
                # Rotation: rotate images/labels together and recompute hv vectors with the rotation matrix.
                if random.random() < self.aug_prob['rotate']:
                    angle = random.uniform(self.rotation_degrees[0], self.rotation_degrees[1])
                    img = self._transform_volume_slices(
                        img,
                        lambda sl: TF.rotate(sl, angle, interpolation=InterpolationMode.BILINEAR),
                    )
                    mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
                    backgrd, others = hv_map[0:1, :, :], hv_map[1:4, :, :]
                    others = TF.rotate(others, angle, interpolation=InterpolationMode.BILINEAR)
                    backgrd = 1.0 - torch.clone(others[0:1, :, :])
                    backgrd = torch.clip(backgrd, 0, 1)
                    hv_map = torch.cat([backgrd, others], dim=0)
                    type_map = TF.rotate(type_map, angle, interpolation=InterpolationMode.NEAREST)

                    angle_rad = -np.deg2rad(angle)
                    cos_a = np.cos(angle_rad)
                    sin_a = np.sin(angle_rad)
                    h_map = hv_map[2, :, :].clone()
                    v_map = hv_map[3, :, :].clone()
                    hv_map[2, :, :] = cos_a * h_map - sin_a * v_map
                    hv_map[3, :, :] = sin_a * h_map + cos_a * v_map

                # Horizontal flip: negate the h component.
                if random.random() < self.aug_prob['hflip']:
                    img = self._transform_volume_slices(img, TF.hflip)
                    mask = TF.hflip(mask)
                    hv_map = TF.hflip(hv_map)
                    hv_map[2, :, :] = -hv_map[2, :, :]
                    type_map = TF.hflip(type_map)

                # Vertical flip: negate the v component.
                if random.random() < self.aug_prob['vflip']:
                    img = self._transform_volume_slices(img, TF.vflip)
                    mask = TF.vflip(mask)
                    hv_map = TF.vflip(hv_map)
                    hv_map[3, :, :] = -hv_map[3, :, :]
                    type_map = TF.vflip(type_map)

                # Scaling: apply affine zoom and multiply hv vector magnitudes by the same scale.
                if random.random() < self.aug_prob['scale']:
                    scale = random.uniform(self.scale_factor_range[0], self.scale_factor_range[1])
                    img = self._transform_volume_slices(
                        img,
                        lambda sl: TF.affine(
                            sl, angle=0, translate=[0, 0], scale=scale, shear=[0, 0],
                            interpolation=InterpolationMode.BILINEAR,
                        ),
                    )
                    mask = TF.affine(mask, angle=0, translate=[0, 0], scale=scale, shear=[0, 0], interpolation=InterpolationMode.NEAREST)
                    hv_map = TF.affine(hv_map, angle=0, translate=[0, 0], scale=scale, shear=[0, 0], interpolation=InterpolationMode.BILINEAR)
                    hv_map[2:4, :, :] = hv_map[2:4, :, :] * scale
                    type_map = TF.affine(type_map, angle=0, translate=[0, 0], scale=scale, shear=[0, 0], interpolation=InterpolationMode.NEAREST)

            if pix_aug:
                # Inversion: enable only when polarity varies across devices (default probability 0).
                if random.random() < self.aug_prob['invert']:
                    img = self._transform_volume_slices(img, TF.invert)

                # Brightness jitter
                if random.random() < self.aug_prob['brightness']:
                    brightness_factor = random.uniform(self.brightness_factor_range[0], self.brightness_factor_range[1])
                    img = self._transform_volume_slices(
                        img,
                        lambda sl: TF.adjust_brightness(sl, brightness_factor),
                    )

                # Contrast jitter
                if random.random() < self.aug_prob['contrast']:
                    contrast_factor = random.uniform(self.contrast_factor_range[0], self.contrast_factor_range[1])
                    img = self._transform_volume_slices(
                        img,
                        lambda sl: TF.adjust_contrast(sl, contrast_factor),
                    )

                # Gaussian blur: adapt kernel size to sigma (2*ceil(3*sigma)+1) to avoid truncating large sigma.
                if random.random() < self.aug_prob['blur']:
                    sigma = random.uniform(self.blur_sigma_range[0], self.blur_sigma_range[1])
                    kernel_s = 2 * int(np.ceil(3.0 * sigma)) + 1
                    img = self._transform_volume_slices(
                        img,
                        lambda sl: TF.gaussian_blur(sl, kernel_size=(kernel_s, kernel_s), sigma=sigma),
                    )

                # Downsampling-upsampling: simulate resolution loss from undersampling / binning.
                if random.random() < self.aug_prob['downscale']:
                    *_, h, w = img.shape
                    down_scale_factor = random.uniform(self.downscale_factor_range[0], self.downscale_factor_range[1])
                    temp_h, temp_w = int(h * down_scale_factor), int(w * down_scale_factor)
                    if temp_h > 0 and temp_w > 0:
                        img = self._transform_volume_slices(
                            img,
                            lambda sl: TF.resize(
                                TF.resize(sl, (temp_h, temp_w), interpolation=InterpolationMode.BILINEAR, antialias=True),
                                (h, w), interpolation=InterpolationMode.BILINEAR, antialias=True,
                            ),
                        )

                # Uneven illumination: a low-frequency multiplicative field around 1.0 (typical brightfield/phase-contrast degradation).
                if random.random() < self.aug_prob['illumination']:
                    *_, h, w = img.shape
                    g = self.illumination_grid
                    strength = random.uniform(self.illumination_strength_range[0], self.illumination_strength_range[1])
                    field = F.interpolate(torch.rand(1, 1, g, g), size=(h, w), mode='bilinear', align_corners=False)[0, 0]
                    field = 1.0 + strength * (field - field.mean())
                    img = torch.clip(img * field, 0.0, 1.0)

                # Additive noise (applied last): choose either Poisson shot noise or Gaussian sensor noise.
                if random.random() < self.aug_prob['noise']:
                    img = torch.clip(img, 0.0, 1.0)
                    if random.random() < 0.5:
                        scale = random.uniform(self.poisson_noise_scale_range[0], self.poisson_noise_scale_range[1])
                        img = torch.poisson(img * scale) / scale
                    else:
                        std = random.uniform(self.gaussian_noise_std_range[0], self.gaussian_noise_std_range[1])
                        img = img + torch.randn_like(img) * std
                    img = torch.clip(img, 0.0, 1.0)

            # ---- Repeat percentile normalization after degradation (matching Cellpose). ----
            if self.normalize_percentile:
                img = self._percentile_normalize(img, *self.norm_percentiles)

        # ---- Restore planar dimensions and arrange outputs (decode large-value masks). ----
        if planar_input:
            img = img.squeeze(1)
        img = img.contiguous()
        mask = mask.contiguous()
        mask = torch.round(mask * 100000).int()
        hv_map = hv_map.contiguous()
        type_map = type_map.contiguous()
        return [img, mask, hv_map, type_map]

    @staticmethod
    def _percentile_normalize(volume: torch.Tensor, low_p: float, high_p: float) -> torch.Tensor:
        """Normalize robustly per channel: map [low_p, high_p] percentiles to [0,1] and clip (input (C, Z, H, W))."""
        out = torch.empty_like(volume)
        for c in range(volume.shape[0]):
            flat = volume[c].flatten()
            lo = torch.quantile(flat, low_p / 100.0)
            hi = torch.quantile(flat, high_p / 100.0)
            out[c] = torch.clip((volume[c] - lo) / (hi - lo + 1e-6), 0.0, 1.0)
        return out

    @staticmethod
    def _transform_volume_slices(volume: torch.Tensor, transform_fn: Callable) -> torch.Tensor:
        """Apply a 2D (C, H, W) transform independently to each z slice in (C, Z, H, W)."""
        return torch.stack(
            [transform_fn(volume[:, z_idx]) for z_idx in range(volume.shape[1])],
            dim=1,
        )

    @staticmethod
    def _resize_input_image(img: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """Resize volumetric (C, Z, H, W) inputs to the patch target size along XY."""
        return F.interpolate(
            img.unsqueeze(0),
            size=(img.shape[1], *target_size),
            mode='trilinear',
            align_corners=False,
        ).squeeze(0)
    
# --- scDataSet Class ---
class scDataSet(Dataset):
    def __init__(self, 
                 hdf5_path: str | Path,
                 group_paths: List[str],
                 apply_augmentations:bool=True,
                 target_size:Tuple[int, int]=(224, 224)):
        super().__init__()
        self.hdf5_path = Path(hdf5_path)
        self.group_paths = group_paths
        self.apply_augmentations = apply_augmentations
        self.input_dims = 3
        self.input_z_size = 10
        self.image_channel_count = 0
        self.target_image_channel = None
        
        self.cell_info = []
        self._get_cell_info()

        if self.apply_augmentations:
            self.rotation_degrees = (-20, 20)
            self.scale_factor_range = (0.85, 1.15)
            self.brightness_factor_range = (0.7, 1.3)
            self.contrast_factor_range = (0.8, 1.2)
        self.target_size = target_size

    def configure_3d_input(self, target_image_channel:list[int]|int|None=None, input_z_size:int=10) -> None:
        """Set target biological channels for 3D single-cell crop loading."""
        if isinstance(target_image_channel, int):
            target_image_channel = [target_image_channel]
        self.target_image_channel = target_image_channel
        self.input_z_size = input_z_size

    def _get_cell_info(self):
        """Retrieve single cell info from HDF5 file."""
        self.cell_info = []
        if not self.hdf5_path.exists():
            return
            
        with h5py.File(self.hdf5_path, 'r') as f:
            for group_path in self.group_paths:
                if group_path in f and 'sc_stack' in f[group_path]:
                    sc_group = f[f'{group_path}/sc_stack']
                    self.input_dims = 3
                    self.image_channel_count = max(
                        self.image_channel_count,
                        len(literal_eval(sc_group.attrs.get('channel_z_sizes', '[]'))))
                    meta_info = sc_group.attrs.get('meta_info', '')
                    meta_info = literal_eval(meta_info) if meta_info else {}
                    num_cells = len(sc_group['cells'].keys())
                    for i in range(num_cells):
                        self.cell_info.append({
                            'h5_path': f"{group_path}/sc_stack",
                            'cell_idx': i,
                            'meta_info': meta_info
                        })

    def __len__(self) -> int:
        return len(self.cell_info)

    def __getitem__(self, index:int, geo_aug:bool=True, pix_aug:bool=True) -> dict:
        info = self.cell_info[index]
        with h5py.File(self.hdf5_path, 'r') as f:
            sc_group = f[info['h5_path']]
            image, image_channel_indices = self._load_cell_volume(sc_group, info['cell_idx'])
            coord = sc_group['coords'][info['cell_idx']]
            cell_type = sc_group['cell_types'][info['cell_idx']]

        image_tensor = self.apply_transforms(image, geo_aug=geo_aug, pix_aug=pix_aug)

        return {
            'image': image_tensor,
            'meta_info': info.get('meta_info', {}),
            'coord': torch.from_numpy(coord).float(),
            'cell_type': torch.from_numpy(cell_type).float(),
            'image_channel_indices': torch.tensor(image_channel_indices, dtype=torch.long),
            'data_index': index,
        }

    def __add__(self, other: 'scDataSet') -> 'scDataSet':
        return self.merge_dataset([self, other])

    @staticmethod
    def merge_dataset(dataset_list:List['scDataSet']) -> 'scDataSet':
        if not dataset_list: return None
        first_dataset = dataset_list[0]
        all_group_paths = [p for ds in dataset_list for p in ds.group_paths]
        
        return scDataSet(
            hdf5_path=first_dataset.hdf5_path,
            group_paths=all_group_paths,
            apply_augmentations=first_dataset.apply_augmentations,
            target_size=first_dataset.target_size
        )

    def _load_cell_volume(self, sc_group, cell_idx:int) -> tuple[np.ndarray, list[int]]:
        """Load one raw-channel single-cell crop and z-resample selected channels."""
        cell_group = sc_group['cells'][f'cell_{cell_idx}']['raw_channels']
        if self.target_image_channel is None:
            target_channels = [0]
        elif isinstance(self.target_image_channel, int):
            target_channels = [self.target_image_channel]
        else:
            target_channels = list(self.target_image_channel)
        channel_arrays = []
        z_sizes = []
        for channel_idx in target_channels:
            channel_image = cell_group[f'channel_{channel_idx}']['image'][:]
            channel_arrays.append(channel_image)
            z_sizes.append(int(channel_image.shape[0]))
        if len(set(z_sizes)) != 1:
            raise ValueError(
                f"Selected target_image_channel {target_channels} have inconsistent single-cell z sizes {z_sizes}."
            )
        z_raw = z_sizes[0]
        image_tensor = torch.from_numpy(np.stack(channel_arrays, axis=0)).float().unsqueeze(0)
        if z_raw != self.input_z_size:
            image_tensor = F.interpolate(
                image_tensor,
                size=(self.input_z_size, image_tensor.shape[-2], image_tensor.shape[-1]),
                mode='trilinear',
                align_corners=False)
        return image_tensor.squeeze(0).numpy().astype(np.float32), target_channels

    def apply_transforms(self, image:np.ndarray, geo_aug:bool=True, pix_aug:bool=True) -> torch.Tensor:
        """Resize and augment one cell image; planar (C, H, W) inputs are handled as Z=1 volumes."""
        planar_input = image.ndim == 3
        max_axes = tuple(range(1, image.ndim))
        img = torch.from_numpy(image / (np.max(image, axis=max_axes, keepdims=True) + 1e-6)).float()
        if planar_input:
            img = img.unsqueeze(1)

        img = GridDataSet._resize_input_image(img, self.target_size)

        if self.apply_augmentations:
            if geo_aug:
                if random.random() < 0.5:
                    angle = random.uniform(*self.rotation_degrees)
                    img = GridDataSet._transform_volume_slices(
                        img,
                        lambda sl: TF.rotate(sl, angle, interpolation=InterpolationMode.NEAREST),
                    )
                if random.random() < 0.5:
                    img = GridDataSet._transform_volume_slices(img, TF.hflip)
                if random.random() < 0.5:
                    img = GridDataSet._transform_volume_slices(img, TF.vflip)
                if random.random() < 0.5:
                    scale = random.uniform(*self.scale_factor_range)
                    img = GridDataSet._transform_volume_slices(
                        img,
                        lambda sl: TF.affine(sl, angle=0, translate=[0, 0], scale=scale, shear=[0, 0]),
                    )
            if pix_aug:
                if random.random() < 0.5:
                    brightness_factor = random.uniform(*self.brightness_factor_range)
                    img = GridDataSet._transform_volume_slices(
                        img,
                        lambda sl: TF.adjust_brightness(sl, brightness_factor),
                    )
                if random.random() < 0.5:
                    contrast_factor = random.uniform(*self.contrast_factor_range)
                    img = GridDataSet._transform_volume_slices(
                        img,
                        lambda sl: TF.adjust_contrast(sl, contrast_factor),
                    )

        if planar_input:
            img = img.squeeze(1)
        return img.contiguous()
    
# GridImageLite Class for quick grid division without heavy augmentations or complex metadata handling, mainly for inference or simple patch extraction.
class GridImageLite:
    def __init__(self,
                 image:np.ndarray|list[np.ndarray],
                 grid_size:Tuple|int=(2, 2),
                 grid_target_size:Tuple|int=(512, 512),
                 margin_ratio:float|tuple|list=0.1,
                 rescale:float=1.0,
                 meta_info:dict|None=None,
                 image_name:str|None=None,
                 image_axis_order:str|list[str]='auto',
                 normalize:bool=False,
                 input_z_size:int=10,
                 ):
        pass

        self.meta_info = {} if meta_info is None else dict(meta_info)
        self.image_name = image_name
        self.grid_size = (grid_size, grid_size) if type(grid_size) is int else grid_size
        self.grid_target_size = (grid_target_size, grid_target_size) if type(grid_target_size) is int else grid_target_size
        self.margin_ratio = margin_ratio
        self.rescale = rescale
        self.image_axis_order = image_axis_order
        self.normalize = normalize

        self.enable_geo_aug = False
        self.enable_pix_aug = False

        image = self._standardize_lite_image_channels(image, image_axis_order=image_axis_order)
        self.source_channel_dtypes = [np.asarray(channel).dtype for channel in image]

        if self.normalize:
            image = GridImage.normalize_image(image)

        self.input_dims = 3
        self.input_z_size = input_z_size
        self.image_channel_count = len(image)
        self.target_image_channel = None

        self.image, patch_h, patch_w, margin_h, margin_w = self.prepare_grid_divide(
            image, self.margin_ratio, self.grid_size, rescale=self.rescale)
        
        self.stitch_info = {
            'patch_h': patch_h,
            'patch_w': patch_w,
            'margin_h': margin_h,
            'margin_w': margin_w,
            'original_h': int(image[0].shape[-2]),
            'original_w': int(image[0].shape[-1])
        }

        self.patches = self.perform_divide()

    def configure_3d_input(self, target_image_channel:list[int]|int|None=None, input_z_size:int=10) -> None:
        """Set the biological channels used for patch z-size checks and z resampling."""
        self.target_image_channel = target_image_channel
        self.input_z_size = input_z_size
        self.target_image_channel = self._resolve_target_channels()

    def _resolve_target_channels(self) -> list[int]:
        """Return dataset channel ids selected for patch loading."""
        if self.target_image_channel is None:
            self.target_image_channel = list(range(self.image_channel_count))
        elif isinstance(self.target_image_channel, int):
            self.target_image_channel = [self.target_image_channel]
        else:
            self.target_image_channel = list(self.target_image_channel)
        return self.target_image_channel

    def get_patch_meta_info(self, row:int, col:int) -> dict:
        """Return image metadata inherited by one grid patch."""
        patch_meta = dict(self.meta_info)
        patch_meta['patch_row'] = row
        patch_meta['patch_col'] = col
        return patch_meta

    def __getitem__(self, key) -> dict[str, torch.Tensor | dict | List[torch.Tensor]]:
        """
        Gets one patch from the grid using a (row, col) integer tuple.

        Args:
            key: `(row, col)` indices into the grid.

        Returns:
            A dictionary with one patch image tensor and its metadata.
        """
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("GridImageLite indexing requires a tuple of two integer indices")

        row_key, col_key = key
        num_rows, num_cols = self.grid_size
        if not isinstance(row_key, int) or not isinstance(col_key, int):
            raise TypeError("GridImageLite indices must be integers")
        if not (0 <= row_key < num_rows and 0 <= col_key < num_cols):
            raise IndexError("GridImageLite index out of range")

        target_channels = self._resolve_target_channels()
        patch_channels = [self.patches[row_key][col_key][channel_idx] for channel_idx in target_channels]
        patch_czyx = self._stack_patch_channels(patch_channels)
        img_tensor = self._normalize_and_resize_patch_czyx(patch_czyx, channel_indices=target_channels)

        return {
            'image': img_tensor,
            'meta_info': self.get_patch_meta_info(row_key, col_key),
            'image_channel_indices': torch.tensor(target_channels, dtype=torch.long),
        }

    def perform_divide(self) -> list[list[list[np.ndarray]]]:
        """Divide channel volumes into a row-major grid of per-patch channel lists."""

        grid_patch_list = [[] for _ in range(self.grid_size[0])]

        patch_h, patch_w = self.stitch_info['patch_h'], self.stitch_info['patch_w']
        margin_h, margin_w = self.stitch_info['margin_h'], self.stitch_info['margin_w']

        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                row_slice = slice((i * (patch_h - margin_h)), ((i + 1) * patch_h - i * margin_h))
                col_slice = slice((j * (patch_w - margin_w)), ((j + 1) * patch_w - j * margin_w))
                grid_patch_list[i].append(
                    [channel[:, row_slice, col_slice] for channel in self.image]
                )
        
        return grid_patch_list

    def _normalize_and_resize_patch_czyx(self, patch_czyx: np.ndarray,
                                         channel_indices:list[int]|None=None) -> torch.Tensor:
        """Z-resample, dtype-scale, and resize one CZYX patch, matching GridDataSet loading."""
        if patch_czyx.ndim != 4:
            raise ValueError(f"Expected CZYX patch, got shape {patch_czyx.shape}.")

        if channel_indices is None:
            channel_indices = list(range(patch_czyx.shape[0]))

        z_raw = patch_czyx.shape[1]
        patch_tensor = torch.from_numpy(patch_czyx).float().unsqueeze(0)
        if z_raw != self.input_z_size:
            patch_tensor = F.interpolate(
                patch_tensor,
                size=(self.input_z_size, patch_tensor.shape[-2], patch_tensor.shape[-1]),
                mode='trilinear',
                align_corners=False,
            )
        patch_czyx = patch_tensor.squeeze(0).numpy().astype(np.float32)

        for tensor_idx, source_channel_idx in enumerate(channel_indices):
            source_dtype = self.source_channel_dtypes[source_channel_idx]
            if np.issubdtype(source_dtype, np.integer):
                patch_czyx[tensor_idx] = patch_czyx[tensor_idx] / (
                    float(np.iinfo(source_dtype).max) + 1e-6
                )

        patch_tensor = torch.from_numpy(patch_czyx).float()
        return GridDataSet._resize_input_image(patch_tensor, self.grid_target_size)

    @staticmethod
    def _standardize_lite_image_channels(image: np.ndarray | list[np.ndarray],
                                         image_axis_order: str | list[str] = 'auto') -> list[np.ndarray]:
        """Normalize inline inference inputs into raw channel volumes shaped (Z, H, W)."""
        images = [image] if isinstance(image, np.ndarray) else list(image)
        axis_orders = [image_axis_order] * len(images) if isinstance(image_axis_order, str) else image_axis_order

        channel_arrays = []
        split_single_file_channels = len(images) == 1
        for image_idx, (arr, axis_order) in enumerate(zip(images, axis_orders)):
            channels, _ = GridImage._standardize_image_array(
                np.asarray(arr),
                source_path=f'inline_array_{image_idx}',
                image_axis_order=axis_order,
                split_channel_axis=split_single_file_channels,
            )
            channel_arrays.extend(channels)
        return channel_arrays

    @staticmethod
    def prepare_grid_divide(channels: list[np.ndarray], margin_ratio:float|tuple|list, 
                            grid_size:tuple[int, int], rescale:float=1.0):
        """Define grid geometry from channel volumes and optionally resize them to fit the grid."""
        image_h, image_w = int(channels[0].shape[-2]), int(channels[0].shape[-1])
        if rescale != 1.0:
            h, w  = int(image_h * rescale), int(image_w * rescale)
        else:
            h, w = image_h, image_w

        if isinstance(margin_ratio, float):
            margin_ratio = (margin_ratio, margin_ratio)

        patch_h = int(h / ( (1 - margin_ratio[0]) * grid_size[0] + margin_ratio[0]))
        patch_w = int(w / ( (1 - margin_ratio[1]) * grid_size[1] + margin_ratio[1]))
        margin_h = int(patch_h * margin_ratio[0])
        margin_w = int(patch_w * margin_ratio[1])

        actual_total_h = (patch_h - margin_h) * grid_size[0] + margin_h
        actual_total_w = (patch_w - margin_w) * grid_size[1] + margin_w

        if actual_total_h != image_h or actual_total_w != image_w:
            target_hw = (actual_total_h, actual_total_w)
            channels = [GridImage._resize_channel_volume_xy(channel, target_hw) for channel in channels]

        return channels, patch_h, patch_w, margin_h, margin_w

    @staticmethod
    def _stack_patch_channels(patch_channels: list[np.ndarray]) -> np.ndarray:
        """Stack cropped (Z, H, W) channel volumes into one (C, Z, H, W) patch array."""
        stacked_channels = []
        for channel_idx, channel in enumerate(patch_channels):
            channel = np.asarray(channel)
            if channel.ndim == 2:
                channel = np.expand_dims(channel, axis=0)
            elif channel.ndim != 3:
                raise ValueError(
                    f"GridImageLite patch channel {channel_idx} must be (Z, H, W) or (H, W), got {channel.shape}."
                )
            stacked_channels.append(channel)
        return np.stack(stacked_channels, axis=0)

