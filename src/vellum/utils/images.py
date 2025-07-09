import pdf2image


__all__ = ['extract_images']


def extract_images(file_name, dpi=500):
    """
    Extract a list of images from a PDF file.
    """
    return pdf2image.convert_from_path(file_name, dpi)
