import qrcode

# Data to be encoded
data = "https://example.com"

# Generate QR code
qr = qrcode.QRCode(
    version=1,  # Version 1 means 21x21 matrix, adjust size with version number
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=50,  # Size of each box in pixels
    border=4,  # Border thickness in boxes, default is 4
)
qr.add_data(data)
qr.make(fit=True)

# Create an image from the QR Code instance
img = qr.make_image(fill_color="black", back_color="white")

# Save it somewhere, adjust path as needed
img.save("assets/qr_code.png")
