import segno
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from io import BytesIO
import os

def generate_micro_qr_codes_pdf(num_codes, qr_sizes_mm):
    """
    Generate a PDF file containing multiple Micro QR codes, each centered on its own page.

    Args:
        num_codes (int): The number of Micro QR codes to generate.
        qr_sizes_mm (list): A list of sizes (in millimeters) for each Micro QR code.
    """
    # Ensure the assets directory exists
    assets_dir = "assets"
    os.makedirs(assets_dir, exist_ok=True)
    
    pdf_path = os.path.join(assets_dir, "micro_qr_code.pdf")
    c = canvas.Canvas(pdf_path, pagesize=A4)

    for i in range(num_codes):
        qr_size_mm = qr_sizes_mm[min(i, len(qr_sizes_mm) - 1)]  # Use the last size for excess codes
        
        # Generate Micro QR code
        qr = segno.make_micro(f'{i}', error='L')
        
        img_buffer = BytesIO()
        qr.save(img_buffer, kind='png', scale=10)  # Adjust scale as needed
        img_buffer.seek(0)
        img_reader = ImageReader(img_buffer)

        # Calculate the QR code size in points and its position
        qr_size_points = qr_size_mm * mm
        page_center_x = A4[0] / 2
        page_center_y = A4[1] / 2
        qr_y_position = page_center_y - (qr_size_points / 2)
        x_position = page_center_x - (qr_size_points / 2)

        # Draw the Micro QR code image
        c.drawImage(img_reader, x_position, qr_y_position, width=qr_size_points, height=qr_size_points)
        
        c.showPage()  # Add a new page for the next Micro QR code

    c.save()  # Save the PDF

def generate_micro_qr_codes_imgs(num_codes, resolution):
    """
    Generate individual PNG images for a specified number of Micro QR codes.

    Args:
        num_codes (int): The number of Micro QR codes to generate.
        resolution (int): The resolution for each QR code image.
    """
    assets_dir = "assets"
    os.makedirs(assets_dir, exist_ok=True)
    
    for i in range(1, num_codes + 1):
        qr = segno.make_micro(f'{i}', error='L')
        img_path = os.path.join(assets_dir, f"micro_qr_code_{i}.png")
        qr.save(img_path, scale=50, kind='png', border=0)  # Adjust scale as needed to fit the desired resolution

# Parameters
num_codes = 4  # Number of QR codes to generate
qr_sizes_mm = [100, 100, 100, 100]  # Sizes in millimeters for each Micro QR code
resolution = 1000  # Desired resolution for PNG images

# Generate QR code images and PDF
generate_micro_qr_codes_imgs(num_codes, resolution)
# generate_micro_qr_codes_pdf(num_codes, qr_sizes_mm)

print("Micro QR code images and PDF generated successfully.")
