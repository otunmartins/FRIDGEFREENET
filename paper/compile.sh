#!/bin/bash

# LaTeX Compilation Script for RevTeX4-2 Document
# This script compiles the apssamp.tex document with proper handling of intermediate files

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MAIN_TEX="apssamp"
OUTPUT_DIR="output"
TEMP_DIR="temp"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to clean intermediate files
clean_intermediate() {
    print_status "Cleaning intermediate files..."
    
    # Standard LaTeX intermediate files
    rm -f *.aux *.log *.bbl *.blg *.toc *.lof *.lot *.fls *.fdb_latexmk
    rm -f *.synctex.gz *.synctex *.pdfsync *.out *.nav *.snm *.vrb
    rm -f *.figlist *.makefile *.figlist.bak *.makefile.bak
    rm -f *.dvi *.ps *.eps *.brf *.mtc* *.slf* *.slt* *.stc*
    
    # RevTeX specific files
    rm -f *.Notes.bib *.end *.eledsec* *.upa *.upb
    
    # Additional intermediate files
    rm -f *.run.xml *.bcf *.idx *.ilg *.ind *.lol *.thm
    rm -f *.ent *.fff *.ttt *.auxlock *.backup
    
    # Temporary directories
    rm -rf $TEMP_DIR
    
    print_success "Intermediate files cleaned"
}

# Function to setup directories
setup_directories() {
    print_status "Setting up directories..."
    mkdir -p $OUTPUT_DIR
    mkdir -p $TEMP_DIR
    print_success "Directories ready"
}

# Function to compile LaTeX
compile_latex() {
    print_status "Starting LaTeX compilation for $MAIN_TEX.tex..."
    
    # Check if main tex file exists
    if [ ! -f "$MAIN_TEX.tex" ]; then
        print_error "Main LaTeX file $MAIN_TEX.tex not found!"
        exit 1
    fi
    
    # First pass
    print_status "Running first pdflatex pass..."
    pdflatex -interaction=nonstopmode -output-directory=$TEMP_DIR "$MAIN_TEX.tex" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        print_error "First pdflatex pass failed. Check $TEMP_DIR/$MAIN_TEX.log for details."
        cat "$TEMP_DIR/$MAIN_TEX.log" | tail -20
        exit 1
    fi
    
    # Check if bibtex is needed
    if grep -q "\\bibliography" "$MAIN_TEX.tex"; then
        print_status "Running bibtex..."
        cd $TEMP_DIR
        bibtex "$MAIN_TEX" > /dev/null 2>&1
        if [ $? -ne 0 ]; then
            print_warning "Bibtex encountered issues, continuing..."
        fi
        cd ..
    fi
    
    # Second pass
    print_status "Running second pdflatex pass..."
    pdflatex -interaction=nonstopmode -output-directory=$TEMP_DIR "$MAIN_TEX.tex" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        print_error "Second pdflatex pass failed. Check $TEMP_DIR/$MAIN_TEX.log for details."
        cat "$TEMP_DIR/$MAIN_TEX.log" | tail -20
        exit 1
    fi
    
    # Third pass (for RevTeX cross-references)
    print_status "Running third pdflatex pass (RevTeX requirements)..."
    pdflatex -interaction=nonstopmode -output-directory=$TEMP_DIR "$MAIN_TEX.tex" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        print_error "Third pdflatex pass failed. Check $TEMP_DIR/$MAIN_TEX.log for details."
        cat "$TEMP_DIR/$MAIN_TEX.log" | tail -20
        exit 1
    fi
    
    # Move final PDF to output directory
    if [ -f "$TEMP_DIR/$MAIN_TEX.pdf" ]; then
        cp "$TEMP_DIR/$MAIN_TEX.pdf" "$OUTPUT_DIR/"
        cp "$TEMP_DIR/$MAIN_TEX.pdf" .  # Also copy to current directory
        print_success "PDF generated successfully: $MAIN_TEX.pdf"
    else
        print_error "PDF generation failed!"
        exit 1
    fi
}

# Function to check LaTeX installation
check_latex() {
    print_status "Checking LaTeX installation..."
    
    if ! command -v pdflatex &> /dev/null; then
        print_error "pdflatex not found! Please install a LaTeX distribution (TeX Live, MiKTeX, etc.)"
        exit 1
    fi
    
    if ! command -v bibtex &> /dev/null; then
        print_warning "bibtex not found! Bibliography compilation may fail."
    fi
    
    print_success "LaTeX installation check passed"
}

# Function to show help
show_help() {
    echo "LaTeX Compilation Script for RevTeX4-2"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -c, --clean         Clean intermediate files only"
    echo "  -f, --force-clean   Clean all files including output"
    echo "  -q, --quick         Quick compile (single pass)"
    echo "  -v, --verbose       Verbose output"
    echo ""
    echo "Examples:"
    echo "  $0                  Compile the document"
    echo "  $0 --clean          Clean intermediate files"
    echo "  $0 --force-clean    Clean all generated files"
    echo ""
}

# Function for quick compile (single pass)
quick_compile() {
    print_status "Quick compile (single pass)..."
    setup_directories
    pdflatex -interaction=nonstopmode -output-directory=$TEMP_DIR "$MAIN_TEX.tex"
    if [ -f "$TEMP_DIR/$MAIN_TEX.pdf" ]; then
        cp "$TEMP_DIR/$MAIN_TEX.pdf" "$OUTPUT_DIR/"
        cp "$TEMP_DIR/$MAIN_TEX.pdf" .
        print_success "Quick compile completed: $MAIN_TEX.pdf"
    else
        print_error "Quick compile failed!"
        exit 1
    fi
}

# Function for verbose compilation
verbose_compile() {
    print_status "Starting verbose LaTeX compilation..."
    setup_directories
    
    # First pass with verbose output
    print_status "Running first pdflatex pass (verbose)..."
    pdflatex -interaction=nonstopmode -output-directory=$TEMP_DIR "$MAIN_TEX.tex"
    
    # Check for bibliography
    if grep -q "\\bibliography" "$MAIN_TEX.tex"; then
        print_status "Running bibtex (verbose)..."
        cd $TEMP_DIR
        bibtex "$MAIN_TEX"
        cd ..
    fi
    
    # Second pass
    print_status "Running second pdflatex pass (verbose)..."
    pdflatex -interaction=nonstopmode -output-directory=$TEMP_DIR "$MAIN_TEX.tex"
    
    # Third pass
    print_status "Running third pdflatex pass (verbose)..."
    pdflatex -interaction=nonstopmode -output-directory=$TEMP_DIR "$MAIN_TEX.tex"
    
    if [ -f "$TEMP_DIR/$MAIN_TEX.pdf" ]; then
        cp "$TEMP_DIR/$MAIN_TEX.pdf" "$OUTPUT_DIR/"
        cp "$TEMP_DIR/$MAIN_TEX.pdf" .
        print_success "Verbose compile completed: $MAIN_TEX.pdf"
    else
        print_error "Verbose compile failed!"
        exit 1
    fi
}

# Main script logic
main() {
    case "${1:-}" in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--clean)
            clean_intermediate
            exit 0
            ;;
        -f|--force-clean)
            clean_intermediate
            rm -rf $OUTPUT_DIR
            rm -f *.pdf
            print_success "All generated files cleaned"
            exit 0
            ;;
        -q|--quick)
            check_latex
            quick_compile
            exit 0
            ;;
        -v|--verbose)
            check_latex
            verbose_compile
            exit 0
            ;;
        "")
            # Default compilation
            check_latex
            setup_directories
            compile_latex
            print_success "Compilation completed successfully!"
            print_status "Output files:"
            echo "  - PDF: $MAIN_TEX.pdf"
            echo "  - Output directory: $OUTPUT_DIR/"
            echo "  - Temp files: $TEMP_DIR/"
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
}

# Trap to clean up on exit
trap 'if [ $? -ne 0 ]; then print_error "Compilation failed! Check logs in $TEMP_DIR/"; fi' EXIT

# Run main function
main "$@" 