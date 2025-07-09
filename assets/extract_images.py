import pdf2image


pages = pdf2image.convert_from_path('devito.pdf', 500)

for count, page in enumerate(pages):
    page.save(f'out{count}.jpg', 'JPEG')