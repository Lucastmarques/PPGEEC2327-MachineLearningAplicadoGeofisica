"""Leitura dos dados oficiais com verificação do pacote original."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import os
import tempfile
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
from scipy.io import loadmat

ARCHIVE_URL = "http://utam.gg.utah.edu/Classes/ML/LAB1/Chapter.Book.Sparse/Chapter.Sparse2/csc_code.zip"
ARCHIVE_SHA256 = "fe5bd09472bd0f2d334450be19fe1f8ce56864b536e3a5d26eaa8fe1fd3fd44f"


def ensure_archive(root, *, download=True):
    """Usa o ZIP existente ou faz download atômico; recusa conteúdo alterado."""
    path = Path(root) / "original" / "csc_code.zip"
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"Pacote ausente: {path}. URL: {ARCHIVE_URL}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(ARCHIVE_URL, timeout=60) as response:
            payload = response.read()
        if sha256(payload).hexdigest() != ARCHIVE_SHA256:
            raise ValueError("SHA-256 do download difere do pacote inspecionado.")
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".partial", delete=False) as file:
            temp = Path(file.name)
            file.write(payload)
        try:
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
    if sha256(path.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError(f"Pacote original alterado: {path}. Verifique-o antes de substituir o arquivo.")
    return path


def load_original_data(root, *, download=True):
    """Lê os MAT em memória, sem executar MATLAB nem sobrescrever a extração."""
    path = ensure_archive(root, download=download)
    with ZipFile(path) as archive:
        gathers = loadmat(BytesIO(archive.read("csc_code/b.mat")), mat_dtype=True)["b"]
        reference = loadmat(BytesIO(archive.read("csc_code/filters.mat")), struct_as_record=False, mat_dtype=True)
    if gathers.shape != (181, 100, 5) or not np.isfinite(gathers).all():
        raise ValueError("Dados sísmicos incompatíveis com o pacote inspecionado.")
    return gathers.astype(np.float64), reference
