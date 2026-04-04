#!/bin/bash

# LaTeX Compilation Script for RevTeX4-2 Document with BibTeX Support
# This script compiles the apssamp.tex document with proper handling of bibliography and intermediate files

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

# Function to compile LaTeX with proper BibTeX handling
compile_latex() {
    print_status "Starting LaTeX compilation for $MAIN_TEX.tex..."
    print_status "This will run the complete pdflatex → bibtex → pdflatex → pdflatex sequence"
    
    # Check if main tex file exists
    if [ ! -f "$MAIN_TEX.tex" ]; then
        print_error "Main LaTeX file $MAIN_TEX.tex not found!"
        exit 1
    fi
    
    # Check if bibliography file exists
    BIB_FILE=""
    if [ -f "references.bib" ]; then
        BIB_FILE="references.bib"
    elif [ -f "$MAIN_TEX.bib" ]; then
        BIB_FILE="$MAIN_TEX.bib"
    fi
    
    if [ -n "$BIB_FILE" ]; then
        print_status "Found bibliography file: $BIB_FILE"
    fi
    
    # First pass - generates .aux file with citation information
    print_status "Step 1/4: Running first pdflatex pass (generating .aux file)..."
    pdflatex -interaction=nonstopmode -output-directory=$TEMP_DIR "$MAIN_TEX.tex" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        print_error "First pdflatex pass failed. Check $TEMP_DIR/$MAIN_TEX.log for details."
        cat "$TEMP_DIR/$MAIN_TEX.log" | tail -20
        exit 1
    fi
    
    # Check if bibtex is needed and run it
    NEEDS_BIBTEX=false
    if grep -q "\\\\bibliography" "$MAIN_TEX.tex"; then
        NEEDS_BIBTEX=true
    elif [ -n "$BIB_FILE" ]; then
        # If we found a bibliography file, assume we need BibTeX even if command not detected
        NEEDS_BIBTEX=true
        print_status "Bibliography file found, enabling BibTeX processing"
    fi
    
    if [ "$NEEDS_BIBTEX" = "true" ]; then
        print_status "Step 2/4: Running BibTeX (processing bibliography)..."
        
        # Copy bibliography file to temp directory if it exists
        if [ -n "$BIB_FILE" ] && [ -f "$BIB_FILE" ]; then
            cp "$BIB_FILE" "$TEMP_DIR/"
            print_status "Copied $BIB_FILE to temp directory"
        fi
        
        cd $TEMP_DIR
        bibtex "$MAIN_TEX" > bibtex.log 2>&1
        BIBTEX_EXIT_CODE=$?
        cd ..
        
        if [ $BIBTEX_EXIT_CODE -eq 0 ]; then
            print_success "BibTeX completed successfully"
            # Check if .bbl file was generated
            if [ -f "$TEMP_DIR/$MAIN_TEX.bbl" ]; then
                BBL_ENTRIES=$(grep -c "bibitem" "$TEMP_DIR/$MAIN_TEX.bbl" 2>/dev/null || echo "0")
                print_status "Generated bibliography with $BBL_ENTRIES entries"
            fi
        else
            print_warning "BibTeX encountered issues. Check $TEMP_DIR/bibtex.log for details."
            if [ -f "$TEMP_DIR/bibtex.log" ]; then
                echo "BibTeX log (last 10 lines):"
                tail -10 "$TEMP_DIR/bibtex.log"
            fi
        fi
    else
        print_status "Step 2/4: Skipping BibTeX (no bibliography file or command found)"
    fi
    
    # Second pass - incorporates bibliography and resolves citations
    print_status "Step 3/4: Running second pdflatex pass (incorporating bibliography)..."
    pdflatex -interaction=nonstopmode -output-directory=$TEMP_DIR "$MAIN_TEX.tex" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        print_error "Second pdflatex pass failed. Check $TEMP_DIR/$MAIN_TEX.log for details."
        cat "$TEMP_DIR/$MAIN_TEX.log" | tail -20
        exit 1
    fi
    
    # Third pass - finalizes cross-references and citations
    print_status "Step 4/4: Running third pdflatex pass (finalizing cross-references)..."
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
        
        # Get PDF info
        PDF_SIZE=$(ls -lh "$MAIN_TEX.pdf" | awk '{print $5}')
        PDF_PAGES=$(pdfinfo "$MAIN_TEX.pdf" 2>/dev/null | grep "Pages:" | awk '{print $2}' || echo "unknown")
        
        print_success "PDF generated successfully: $MAIN_TEX.pdf ($PDF_SIZE, $PDF_PAGES pages)"
        
        # Check for citation warnings in the log
        if [ -f "$TEMP_DIR/$MAIN_TEX.log" ]; then
            UNDEFINED_CITATIONS=$(grep -c "Citation.*undefined\|natbib.*undefined" "$TEMP_DIR/$MAIN_TEX.log" 2>/dev/null || echo "0")
            # Ensure we have a valid number
            if ! [[ "$UNDEFINED_CITATIONS" =~ ^[0-9]+$ ]]; then
                UNDEFINED_CITATIONS=0
            fi
            if [ "$UNDEFINED_CITATIONS" -gt 0 ]; then
                print_warning "Found $UNDEFINED_CITATIONS undefined citations. Check the log file."
            else
                print_success "All citations resolved successfully!"
            fi
        fi
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
    else
        print_success "BibTeX found - bibliography support enabled"
    fi
    
    if command -v pdfinfo &> /dev/null; then
        print_success "pdfinfo found - PDF details will be shown"
    fi
    
    print_success "LaTeX installation check passed"
}

# Function to show help
show_help() {
    echo "LaTeX Compilation Script for RevTeX4-2 with BibTeX Support"
    echo ""
    echo "This script runs the complete LaTeX compilation sequence:"
    echo "  1. pdflatex (first pass) - generates .aux file"
    echo "  2. bibtex - processes bibliography"
    echo "  3. pdflatex (second pass) - incorporates bibliography"
    echo "  4. pdflatex (third pass) - finalizes cross-references"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -c, --clean         Clean intermediate files only"
    echo "  -f, --force-clean   Clean all files including output"
    echo "  -q, --quick         Quick compile (single pass, no bibliography)"
    echo "  -v, --verbose       Verbose output"
    echo ""
    echo "Examples:"
    echo "  $0                  Full compile with bibliography"
    echo "  $0 --clean          Clean intermediate files"
    echo "  $0 --force-clean    Clean all generated files"
    echo "  $0 --quick          Quick single-pass compile"
    echo ""
}

# Function for quick compile (single pass)
quick_compile() {
    print_status "Quick compile (single pass, no bibliography processing)..."
    setup_directories
    pdflatex -interaction=nonstopmode -output-directory=$TEMP_DIR "$MAIN_TEX.tex"
    if [ -f "$TEMP_DIR/$MAIN_TEX.pdf" ]; then
        cp "$TEMP_DIR/$MAIN_TEX.pdf" "$OUTPUT_DIR/"
        cp "$TEMP_DIR/$MAIN_TEX.pdf" .
        print_success "Quick compile completed: $MAIN_TEX.pdf"
        print_warning "Note: Bibliography not processed in quick mode"
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
    NEEDS_BIBTEX=false
    if grep -q "\\\\bibliography" "$MAIN_TEX.tex"; then
        NEEDS_BIBTEX=true
    elif [ -f "references.bib" ] || [ -f "$MAIN_TEX.bib" ]; then
        NEEDS_BIBTEX=true
    fi
    
    if [ "$NEEDS_BIBTEX" = "true" ]; then
        print_status "Running bibtex (verbose)..."
        if [ -f "references.bib" ]; then
            cp "references.bib" "$TEMP_DIR/"
        elif [ -f "$MAIN_TEX.bib" ]; then
            cp "$MAIN_TEX.bib" "$TEMP_DIR/"
        fi
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
            # Default compilation with full BibTeX support
            check_latex
            setup_directories
            compile_latex
            print_success "Compilation completed successfully!"
            print_status "Output files:"
            echo "  - PDF: $MAIN_TEX.pdf"
            echo "  - Output directory: $OUTPUT_DIR/"
            echo "  - Temp files: $TEMP_DIR/"
            echo ""
            echo "To clean intermediate files: $0 --clean"
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