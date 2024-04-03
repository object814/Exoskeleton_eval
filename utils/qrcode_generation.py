import qrcode
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from io import BytesIO

def generate_qr_codes_pdf(num_codes, qr_sizes_mm):
    """
    Generate a PDF file containing multiple QR codes.

    Args:
        num_codes (int): The number of QR codes to generate.
        qr_sizes_mm (list): A list of sizes (in millimeters) for each QR code.

    Generated PDF files will be saved to assets
    """
    c = canvas.Canvas(f"assets/qr_code.pdf", pagesize=A4)  # Create a new PDF in A4 size

    if len(qr_sizes_mm) != num_codes:
        print("Size list does not match the number of QR codes; defaulting to first size for all.")
        qr_sizes_mm = [qr_sizes_mm[0]] * num_codes

    for i, qr_size_mm in enumerate(qr_sizes_mm, start=1):
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(f'{i}')
        qr.make(fit=True)

        # Generate QR code as an image object
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert the QR code image object to a format that ReportLab can use
        img_buffer = BytesIO()
        img.save(img_buffer, format="PNG")
        img_buffer.seek(0)

        # Use ImageReader to wrap the img_buffer
        img_reader = ImageReader(img_buffer)

        # Calculate the QR code size in points
        qr_size_points = qr_size_mm * 2.83465
        
        # Calculate the position for the QR code to be centered on the page
        x_position = (A4[0] - qr_size_points) / 2
        y_position = (A4[1] - qr_size_points) / 2
        
        # Draw the QR code image on the PDF, centered
        c.drawImage(img_reader, x_position, y_position, width=qr_size_points, height=qr_size_points)
        
        # Add a new page for the next QR code
        c.showPage()
        
    c.save()  # Save the PDF

def generate_qr_codes_imgs(num_codes, resolution):
    for i in range(1, num_codes + 1):
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(f'{i}')
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        img = img.resize((resolution, resolution))
        img.save(f"assets/qr_code_{i}.png")

num_codes = 5  # Number of QR codes to generate
resolution = 1000  # Resolution of each QR codeW
qr_sizes_mm = [100, 200, 50, 100, 200]  # Varying sizes for each QR code in millimeters
generate_qr_codes_imgs(num_codes, resolution)
generate_qr_codes_pdf(num_codes, qr_sizes_mm)