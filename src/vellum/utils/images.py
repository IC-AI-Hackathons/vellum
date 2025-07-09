import pdf2image

__all__ = ["extract_images"]


def extract_images(file_name, dpi=500):
    """
    Extract image data from a PDF file.
    """
    pages = pdf2image.convert_from_path(file_name, dpi)

    return pages

