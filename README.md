# ocr-in-database
extract data from images and putting in data base

# Smart OCR Image Data Extractor

A sophisticated Python application that extracts key-value pairs from images (receipts, forms, invoices) using OCR technology and machine learning to continuously improve extraction accuracy.

## 🚀 Features

- **Advanced OCR Processing**: Multiple preprocessing techniques and OCR configurations for optimal text extraction
- **Intelligent Key-Value Extraction**: Three-strategy approach (colon-based, pattern-based, proximity-based)
- **Machine Learning Integration**: Learns from user feedback to improve field mapping accuracy
- **Multiple Image Format Support**: JPG, PNG, BMP, TIFF, etc.
- **Excel Output**: Structured data export with formatting
- **Comprehensive Logging**: Detailed logs for debugging and monitoring
- **Debug Information**: Saves extraction details for analysis

## 📋 Expected Output Format

For receipt/form images, the extractor identifies fields like:

| Field | Value |
|-------|-------|
| Unit | AGRA |
| Location | AGRA |
| GSTIN | 09AADCA0275H1ZU |
| PAN | AADCA0275H |
| Receipt No. | AGR-2425-0004042 |
| Receipt Date | 25-Dec-2024 |
| Agency Code | AM15AMA10 |
| Agency Name | AMAR UJALA [AGRA] |
| Amount | Rs. 6,300.00 |
| Paymode | RTGS/NEFT T/F |
| Remarks | UPI-ANIL KUMAR KUKREJA... |

## 🛠️ Installation

### Prerequisites

1. **Python 3.8+** installed on your system
2. **Tesseract OCR** installed:
   - Windows: Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - Ubuntu: `sudo apt install tesseract-ocr`
   - macOS: `brew install tesseract`

### Setup

1. Clone or download this project
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify Tesseract installation:
   ```bash
   tesseract --version
   ```

## 🚀 Usage

### Basic Usage

1. Place your image files in a folder
2. Run the extractor:
   ```bash
   python smart_ocr_extractor.py --input /path/to/images --output results.xlsx
   ```

### Command Line Options

```bash
python smart_ocr_extractor.py [OPTIONS]

Options:
  -i, --input DIR     Input directory containing images (default: current directory)
  -o, --output FILE   Output Excel file name (default: extracted_data.xlsx)
  -f, --feedback      Enable interactive feedback collection for ML learning
  -h, --help          Show help message
```

### Examples

Extract from current directory:
```bash
python smart_ocr_extractor.py
```

Extract from specific folder:
```bash
python smart_ocr_extractor.py --input C:\Users\Documents\Receipts --output receipts_data.xlsx
```

Enable feedback collection for learning:
```bash
python smart_ocr_extractor.py --feedback
```

## 🧠 Machine Learning Features

### Continuous Learning
The system learns from user corrections to improve future extractions:

1. **Feedback Collection**: Interactive session to correct extracted values
2. **Model Training**: Automatically retrains when sufficient feedback is collected
3. **Field Prediction**: Uses trained models to improve field mapping
4. **Persistence**: Saves models and feedback data for future sessions

### Providing Feedback

When you run with `--feedback` flag, the system will:
1. Show extracted data for each image
2. Ask for corrections to any incorrect values
3. Learn from your corrections
4. Improve future extractions automatically

## 🏗️ Architecture

### Core Components

1. **OCRPreprocessor**: 
   - CLAHE enhancement
   - Bilateral filtering
   - Adaptive thresholding
   - Morphological operations
   - Noise reduction

2. **TextExtractor**:
   - Multiple Tesseract configurations
   - Quality scoring of OCR results
   - Best result selection

3. **KeyValueExtractor**:
   - Colon-based pair extraction
   - Pattern matching for specific data types
   - Proximity-based field detection
   - Duplicate removal and merging

4. **MLLearningSystem**:
   - TF-IDF text vectorization
   - Logistic regression classification
   - Model persistence with joblib
   - Feedback data management

## 📁 Output Files

- **extracted_data.xlsx**: Main results in Excel format
- **debug_[image_name].json**: Debug information for each image
- **ocr_extractor.log**: Application logs
- **ml_models/**: Trained machine learning models
- **ml_models/feedback_data.json**: User feedback history

## 🔧 Configuration

The system automatically detects Tesseract installation. If needed, you can manually configure the path in the `TextExtractor` class.

Common Tesseract paths:
- Windows: `C:\Program Files\Tesseract-OCR\tesseract.exe`
- Linux: `/usr/bin/tesseract`
- macOS: `/usr/local/bin/tesseract`

## 📊 Performance Tips

1. **Image Quality**: Use high-resolution, well-lit images for best results
2. **Image Format**: PNG and TIFF generally work better than JPG
3. **Text Clarity**: Ensure text is clearly visible and not skewed
4. **Feedback**: Provide corrections to improve ML accuracy over time

## 🐛 Troubleshooting

### Common Issues

1. **"Tesseract not found"**:
   - Ensure Tesseract is properly installed
   - Check that tesseract.exe is in your system PATH

2. **"No text extracted"**:
   - Check image quality and resolution
   - Ensure image contains readable text
   - Try different image preprocessing

3. **Poor extraction quality**:
   - Provide feedback to train the ML system
   - Check image preprocessing settings
   - Verify image is not rotated or skewed

### Debug Information

Check the generated debug files:
- `debug_[image].json`: Contains OCR details and extraction process
- `ocr_extractor.log`: Application logs with detailed information

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source. Feel free to use and modify according to your needs.

## 🔮 Future Enhancements

- [ ] Support for table extraction
- [ ] Integration with cloud OCR services
- [ ] Web interface for easier usage
- [ ] Support for handwritten text
- [ ] Batch processing optimization
- [ ] Custom field templates
- [ ] Multi-language support

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the log files
3. Check existing issues
4. Create a new issue with detailed information

---

**Happy Extracting! 🎯**
