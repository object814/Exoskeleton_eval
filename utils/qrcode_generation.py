import qrcode

def generate_qr_codes(num_codes, resolution):
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
resolution = 1000  # Resolution of each QR code
generate_qr_codes(num_codes, resolution)