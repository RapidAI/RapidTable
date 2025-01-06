import io
from pathlib import Path
from typing import Optional, Union

import requests
from tqdm import tqdm

from .logger import get_logger

logger = get_logger("DownloadModel")

CUR_DIR = Path(__file__).resolve()
PROJECT_DIR = CUR_DIR.parent


class DownloadModel:
    cur_dir = PROJECT_DIR

    @staticmethod
    def get_model_path(
        model_type: str, sub_file_type: str, path: Union[str, Path, None]
    ) -> str:
        if path is not None:
            return path

        model_url = KEY_TO_MODEL_URL.get(model_type, {}).get(sub_file_type, None)
        if model_url:
            model_path = DownloadModel.download(model_url)
            return model_path

        logger.info("model url is None, using the default download model %s", path)
        return path

    @classmethod
    def download(cls, model_full_url: Union[str, Path]) -> str:
        save_dir = cls.cur_dir / "models"
        save_dir.mkdir(parents=True, exist_ok=True)

        model_name = Path(model_full_url).name
        save_file_path = save_dir / model_name
        if save_file_path.exists():
            logger.debug("%s already exists", save_file_path)
            return str(save_file_path)

        try:
            logger.info("Download %s to %s", model_full_url, save_dir)
            file = cls.download_as_bytes_with_progress(model_full_url, model_name)
            cls.save_file(save_file_path, file)
        except Exception as exc:
            raise DownloadModelError from exc
        return str(save_file_path)

    @staticmethod
    def download_as_bytes_with_progress(
        url: Union[str, Path], name: Optional[str] = None
    ) -> bytes:
        resp = requests.get(str(url), stream=True, allow_redirects=True, timeout=180)
        total = int(resp.headers.get("content-length", 0))
        bio = io.BytesIO()
        with tqdm(
            desc=name, total=total, unit="b", unit_scale=True, unit_divisor=1024
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=65536):
                pbar.update(len(chunk))
                bio.write(chunk)
        return bio.getvalue()

    @staticmethod
    def save_file(save_path: Union[str, Path], file: bytes):
        with open(save_path, "wb") as f:
            f.write(file)


class DownloadModelError(Exception):
    pass
