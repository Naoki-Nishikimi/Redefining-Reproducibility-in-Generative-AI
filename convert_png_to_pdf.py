from PIL import Image
import glob

# PNG → PDF 変換
for png_file in glob.glob("*.png"):
    img = Image.open(png_file).convert("RGB")
    pdf_file = png_file.replace(".png", ".pdf")
    img.save(pdf_file, "PDF")
    print(f"Converted: {png_file} → {pdf_file}")

print("All PNG images converted to PDF!")
