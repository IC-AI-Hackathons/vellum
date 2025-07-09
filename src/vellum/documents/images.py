import pdf2image
from PIL.Image import Image


__all__ = ['extract_images', 'split_document_pages']


def extract_images(file_name: str, dpi: int = 500) -> list[Image]:
    """
    Extract a list of images from a PDF file.
    """
    return pdf2image.convert_from_path(file_name, dpi)


def split_document_pages(file_name: str, dpi: int = 500) -> list[str]:
    """
    Split a PDF document into a list of images and return their URIs
    """
    images = extract_images(file_name, dpi)

    path, base_name = file_name.rsplit('/', 1)
    file_names = [f'{path}/pages/{base_name}_{i}.png' for i in range(len(images))]
    for i, image in enumerate(images):
        image.save(file_names[i], 'PNG')

    return file_names
