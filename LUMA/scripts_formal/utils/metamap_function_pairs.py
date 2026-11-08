"""Dataset-specific forward/inverse parsers for METAMAP_REGISTRY.

Edit this file to add or change image-name parsers. elements_data.METAMAP
only wraps these pairs and does not hard-code dataset rules.
"""

from datetime import datetime, timedelta
from pathlib import Path
import re


# Frame layout of the archived time-lapse series parsed by this module.
_TIMESERIES_IMAGE_PATTERN = (
    r"Timelapse-(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-RI_MIP\.tif$"
)

# Directory that holds every frame of the time-lapse series. The timepoint-index
# parser has to rank one frame against all of them, so it needs the folder.
# Set it with ``set_timeseries_frame_dir``; when it is None the parser falls back
# to the folder containing the image name it was given.
TIMESERIES_FRAME_DIR: Path | None = None


def set_timeseries_frame_dir(path: str | Path | None) -> Path | None:
    """Point ``METAMAP_imagename_to_timepoint_index`` at the frame folder.

    Args:
        path: Directory holding all ``Timelapse-*-RI_MIP.tif`` frames, or None to clear.

    Returns:
        The resolved directory, or None when the setting was cleared.
    """

    global TIMESERIES_FRAME_DIR
    TIMESERIES_FRAME_DIR = None if path is None else Path(path).expanduser().resolve()
    return TIMESERIES_FRAME_DIR


def _resolve_timeseries_frame_dir(image_name: str | None) -> Path:
    """Resolve the frame directory from the module setting or the image path."""

    if TIMESERIES_FRAME_DIR is not None:
        return TIMESERIES_FRAME_DIR
    if image_name:
        candidate = Path(image_name).expanduser().parent
        if candidate.is_dir():
            return candidate.resolve()
    raise RuntimeError(
        "timepoint index parsing needs the folder that contains all time-lapse frames; "
        "call set_timeseries_frame_dir('/path/to/frames') first, or pass full image paths "
        "whose parent folder holds the frames. Prefer an explicit meta_map dictionary "
        "({image_name: frame_index}) when the frames are not in one shared folder."
    )


def _sorted_timeseries_time_keywords(image_name: str | None) -> list[str]:
    """Rank every matching frame in the resolved folder by acquisition time."""

    frame_dir = _resolve_timeseries_frame_dir(image_name)
    time_keywords = []
    for path in frame_dir.glob('*.tif'):
        match = re.search(_TIMESERIES_IMAGE_PATTERN, path.name)
        if match is None:
            continue
        time_keywords.append(
            datetime(*(int(part) for part in match.groups())).strftime('%Y-%m-%d-%H-%M-%S')
        )
    if not time_keywords:
        raise RuntimeError(
            f"no frames matching 'Timelapse-*-RI_MIP.tif' were found in {frame_dir}; "
            "set the folder with set_timeseries_frame_dir(...) or build meta_info from "
            "an explicit meta_map dictionary."
        )
    return sorted(time_keywords)


def ffov_imagename_to_elapsed_seconds(image_name):
    """Parse one time-series image name and return elapsed seconds."""
    match = re.search(
        r"Timelapse-(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-RI_MIP\.tif$",
        Path(image_name).name,
    )
    frame_time = datetime(*(int(part) for part in match.groups()))
    origin_time = datetime(2026, 1, 28, 2, 9, 57)
    return (frame_time - origin_time).total_seconds()


def finv_elapsed_seconds_to_imagename_time(meta_value):
    """Map elapsed seconds back to this dataset's time keyword."""
    origin_time = datetime(2026, 1, 28, 2, 9, 57)
    return (origin_time + timedelta(seconds=float(meta_value))).strftime('%Y-%m-%d-%H-%M-%S')


def ffov_imagename_to_elapsed_minutes(image_name):
    """Parse one time-series image name and return elapsed minutes."""
    match = re.search(
        r"Timelapse-(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-RI_MIP\.tif$",
        Path(image_name).name,
    )
    frame_time = datetime(*(int(part) for part in match.groups()))
    origin_time = datetime(2026, 1, 28, 2, 9, 57)
    return (frame_time - origin_time).total_seconds() / 60.0


def finv_elapsed_minutes_to_imagename_time(meta_value):
    """Map elapsed minutes back to this dataset's time keyword."""
    origin_time = datetime(2026, 1, 28, 2, 9, 57)
    return (origin_time + timedelta(minutes=float(meta_value))).strftime('%Y-%m-%d-%H-%M-%S')


def ffov_imagename_to_timepoint_index(image_name):
    """Parse one time-series image name and return its sorted frame index."""
    match = re.search(_TIMESERIES_IMAGE_PATTERN, Path(image_name).name)
    if match is None:
        raise ValueError(f"image name {image_name!r} is not a Timelapse-*-RI_MIP.tif frame")
    time_keyword = datetime(*(int(part) for part in match.groups())).strftime('%Y-%m-%d-%H-%M-%S')
    return _sorted_timeseries_time_keywords(image_name).index(time_keyword)


def finv_timepoint_index_to_imagename_time(meta_value):
    """Map one sorted frame index back to this dataset's time keyword."""
    return _sorted_timeseries_time_keywords(None)[int(meta_value)]


def ffov_livecell_imagename_to_day(image_name):
    """Parse one LiveCell image name and return elapsed days."""
    pattern = r'(?P<bio>.+?)_Phase_.+?_(?P<day>\d+)d(?P<hour>\d+)h(?P<minute>\d+)m_\d+\.tif$'
    match = re.match(pattern, Path(image_name).name)
    return int(match.group('day'))


def finv_livecell_day_to_imagename_keyword(meta_value):
    """Map elapsed days back to the LiveCell day keyword."""
    return f'{int(meta_value):02d}d'


def ffov_livecell_imagename_to_hour(image_name):
    """Parse one LiveCell image name and return elapsed hours within the day."""
    pattern = r'(?P<bio>.+?)_Phase_.+?_(?P<day>\d+)d(?P<hour>\d+)h(?P<minute>\d+)m_\d+\.tif$'
    match = re.match(pattern, Path(image_name).name)
    return int(match.group('hour'))


def finv_livecell_hour_to_imagename_keyword(meta_value):
    """Map elapsed hours back to the LiveCell hour keyword."""
    return f'{int(meta_value):02d}h'


def ffov_livecell_imagename_to_minute(image_name):
    """Parse one LiveCell image name and return elapsed minutes within the hour."""
    pattern = r'(?P<bio>.+?)_Phase_.+?_(?P<day>\d+)d(?P<hour>\d+)h(?P<minute>\d+)m_\d+\.tif$'
    match = re.match(pattern, Path(image_name).name)
    return int(match.group('minute'))


def finv_livecell_minute_to_imagename_keyword(meta_value):
    """Map elapsed minutes back to the LiveCell minute keyword."""
    return f'{int(meta_value):02d}m'


def ffov_livecell_imagename_to_biological_type(image_name):
    """Parse one LiveCell image name and return the biological type code."""
    pattern = r'(?P<bio>.+?)_Phase_.+?_(?P<day>\d+)d(?P<hour>\d+)h(?P<minute>\d+)m_\d+\.tif$'
    biological_type_to_code = {
        'A172': 0,
        'BT474': 1,
        'BV2': 2,
        'Huh7': 3,
        'MCF7': 4,
        'SHSY5Y': 5,
        'SKOV3': 6,
        'SkBr3': 7,
    }
    match = re.match(pattern, Path(image_name).name)
    return biological_type_to_code[match.group('bio')]


def finv_livecell_biological_type_to_imagename_keyword(meta_value):
    """Map one LiveCell biological type code back to its name keyword."""
    code_to_biological_type = {
        0: 'A172',
        1: 'BT474',
        2: 'BV2',
        3: 'Huh7',
        4: 'MCF7',
        5: 'SHSY5Y',
        6: 'SKOV3',
        7: 'SkBr3',
    }
    return code_to_biological_type[int(meta_value)]


METAMAP_FUNCTION_PAIRS = {
    'METAMAP_imagename_to_elapsed_seconds': (
        ffov_imagename_to_elapsed_seconds,
        finv_elapsed_seconds_to_imagename_time,
    ),
    'METAMAP_imagename_to_elapsed_minutes': (
        ffov_imagename_to_elapsed_minutes,
        finv_elapsed_minutes_to_imagename_time,
    ),
    'METAMAP_imagename_to_timepoint_index': (
        ffov_imagename_to_timepoint_index,
        finv_timepoint_index_to_imagename_time,
    ),
    'METAMAP_livecell_imagename_to_day': (
        ffov_livecell_imagename_to_day,
        finv_livecell_day_to_imagename_keyword,
    ),
    'METAMAP_livecell_imagename_to_hour': (
        ffov_livecell_imagename_to_hour,
        finv_livecell_hour_to_imagename_keyword,
    ),
    'METAMAP_livecell_imagename_to_minute': (
        ffov_livecell_imagename_to_minute,
        finv_livecell_minute_to_imagename_keyword,
    ),
    'METAMAP_livecell_imagename_to_biological_type': (
        ffov_livecell_imagename_to_biological_type,
        finv_livecell_biological_type_to_imagename_keyword,
    ),
}
