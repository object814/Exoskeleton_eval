import qrcode
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from io import BytesIO

def generate_qr_codes_pdf(num_codes, qr_sizes_mm, gap_mm=20):
    """
    Generate a PDF file containing multiple QR codes each with a black square outline.
    The QR code and the square are aligned horizontally and mirrored along the centerline of the A4 paper
    with a gap between them.

    Args:
        num_codes (int): The number of QR codes to generate.
        qr_sizes_mm (list): A list of sizes (in millimeters) for each QR code.
        gap_mm (int): The gap in millimeters between the QR code and the square.

    Generated PDF files will be saved to assets
    """
    c = canvas.Canvas(f"assets/qr_code.pdf", pagesize=A4)  # Create a new PDF in A4 size

    if len(qr_sizes_mm) != num_codes:
        print("Size list does not match the number of QR codes; defaulting to first size for all.")
        qr_sizes_mm = [qr_sizes_mm[0]] * num_codes

    # Calculate the gap in points
    gap = gap_mm * mm

    for i, qr_size_mm in enumerate(qr_sizes_mm):
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=0,  # Set border to 0
        )
        qr.add_data(f'{i}')
        qr.make(fit=True)

        # Generate QR code as an image object
        img = qr.make_image(fill_color="black", back_color="white")

        # Convert the QR code image object to a format that ReportLab can use
        img_buffer = BytesIO()
        img.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        img_reader = ImageReader(img_buffer)

        # Calculate the QR code size in points
        qr_size_points = qr_size_mm * mm

        # Center positions
        page_center_x = A4[0] / 2
        page_center_y = A4[1] / 2

        # Calculate y positions for QR and square, including the gap
        qr_y_position = page_center_y - qr_size_points - gap / 2
        square_y_position = page_center_y + gap / 2

        # Calculate x position for both square and QR code to be vertically aligned
        x_position = page_center_x - (qr_size_points / 2)

        # Draw the square outline
        c.rect(x_position, square_y_position, qr_size_points, qr_size_points, fill=0)

        # Draw the QR code image
        c.drawImage(img_reader, x_position, qr_y_position, width=qr_size_points, height=qr_size_points)

        # Add a new page for the next QR code
        c.showPage()

    c.save()  # Save the PDF

def generate_qr_codes_imgs(num_codes, resolution):
    for i in range(1, num_codes + 1):
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=0,  # Set border to 0
        )
        qr.add_data(f'{i}')
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        img = img.resize((resolution, resolution))
        img.save(f"assets/qr_code_{i}.png")

num_codes = 2  # Number of QR codes to generate
resolution = 1000  # Resolution of each QR code
qr_sizes_mm = [80,40]  # Varying sizes for each QR code in millimeters
generate_qr_codes_imgs(num_codes, resolution)
generate_qr_codes_pdf(num_codes, qr_sizes_mm)