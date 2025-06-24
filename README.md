# **Doc-VLMs-exp**

An experimental document-focused Vision-Language Model application that provides advanced document analysis, text extraction, and multimodal understanding capabilities. This application features a streamlined Gradio interface for processing both images and videos using state-of-the-art vision-language models specialized in document understanding.

## Features

- **Document-Specialized Models**: Three cutting-edge VLM models optimized for document processing
- **Image Processing**: Advanced document analysis and text extraction from images
- **Video Processing**: Frame-by-frame video analysis with temporal understanding
- **Real-time Streaming**: Live output generation with immediate feedback
- **Dual Output Format**: Raw text stream and formatted markdown results
- **Canvas-Style Interface**: Clean, professional output display
- **Advanced Parameter Control**: Fine-tune generation settings for optimal results

## Supported Models

### DREX-062225-exp (Default)
The Document Retrieval and Extraction Expert model is a specialized fine-tuned version of DocScopeOCR-7B, optimized for document retrieval, content extraction, and analysis recognition. Built on top of the Qwen2.5-VL architecture for superior document understanding.

### VIREX-062225-exp
The Video Information Retrieval and Extraction Expert model is a fine-tuned version of Qwen2.5-VL-7B-Instruct, specifically optimized for advanced video understanding, image comprehension, reasoning, and natural language decision-making through Chain-of-Thought reasoning.

### olmOCR-7B-0225
Based on Qwen2-VL-7B, this model is optimized for document-level optical character recognition (OCR), long-context vision-language understanding, and accurate image-to-text conversion with mathematical LaTeX formatting. Designed with a focus on high-fidelity visual-textual comprehension.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/PRITHIVSAKTHIUR/Doc-VLMs-exp.git
cd Doc-VLMs-exp
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- torch
- transformers
- gradio
- spaces
- numpy
- PIL (Pillow)
- opencv-python

## Usage

### Running the Application

```bash
python app.py
```

The application will launch a Gradio interface accessible through your web browser.

### Image Processing

1. Navigate to the "Image Inference" tab
2. Enter your query in the text field
3. Upload an image file
4. Select your preferred model (DREX-062225-exp recommended for documents)
5. Adjust advanced parameters if needed
6. Click Submit to process

### Video Processing

1. Navigate to the "Video Inference" tab
2. Enter your query describing what you want to analyze
3. Upload a video file
4. Select your preferred model (VIREX-062225-exp recommended for videos)
5. Configure generation parameters as needed
6. Click Submit to process

The application extracts 10 evenly spaced frames from videos and processes them with timestamp information for comprehensive analysis.

### Advanced Configuration

Customize the generation process with these parameters:

- **Max New Tokens**: Control output length (1-2048 tokens)
- **Temperature**: Adjust creativity/randomness (0.1-4.0)
- **Top-p**: Nucleus sampling threshold (0.05-1.0)
- **Top-k**: Top-k sampling limit (1-1000)
- **Repetition Penalty**: Reduce repetitive output (1.0-2.0)

## Example Queries

### Document Processing
- "Convert this page to doc [text] precisely."
- "Extract all text from this document."
- "Analyze the structure of this document."
- "Convert chart to OTSL."
- "Identify key information in this form."

### Video Analysis
- "Explain the video in detail."
- "Describe what happens in this advertisement."
- "Analyze the content of this presentation."
- "Extract text visible in the video frames."

## Output Formats

The application provides two output formats:

1. **Raw Output Stream**: Real-time text generation as it happens
2. **Formatted Result**: Clean markdown formatting for better readability

## Technical Details

### Model Architecture
All models are based on the Qwen2.5-VL and Qwen2-VL architectures, fine-tuned for specific document and video understanding tasks.

### GPU Acceleration
The application automatically uses CUDA when available, with automatic fallback to CPU processing.

### Video Processing Pipeline
Videos are processed by:
1. Extracting 10 evenly distributed frames
2. Converting frames to PIL images
3. Adding timestamp information
4. Processing frames sequentially with the selected model

### Memory Optimization
Models are loaded with 16-bit precision (float16) to optimize memory usage while maintaining performance.

## Hardware Requirements

- **Minimum**: 8GB RAM, modern CPU
- **Recommended**: 16GB+ RAM, CUDA-compatible GPU with 8GB+ VRAM
- **Storage**: 15GB+ free space for model downloads

## Model Selection Guide

- **For Document Analysis**: Use DREX-062225-exp for best results with text extraction and document structure analysis
- **For Video Content**: Use VIREX-062225-exp for comprehensive video understanding and reasoning
- **For OCR Tasks**: Use olmOCR-7B-0225 for high-accuracy text recognition and LaTeX formatting

## Performance Tips

1. Use GPU acceleration when available for faster processing
2. Adjust max_new_tokens based on expected output length
3. Lower temperature values for more focused, factual outputs
4. Higher temperature values for more creative responses
5. Use appropriate models for specific tasks (document vs video)

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is open source. Please check individual model licenses for specific usage terms.

## Acknowledgments

- Hugging Face Transformers library
- Gradio for the user interface framework
- Qwen2.5-VL and Qwen2-VL model teams
- Allen Institute for AI (olmOCR model)

## Support

For bug reports and feature requests, please visit the GitHub repository or use the "Report Bug" link in the application interface.
