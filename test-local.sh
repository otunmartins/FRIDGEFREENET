#!/bin/bash
# Local Testing Script for Insulin-AI Docker Setup
# This script tests the application locally before vast.ai deployment

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOCKER_IMAGE_NAME="insulin-ai-local"
DOCKER_TAG="test"
CONTAINER_NAME="insulin-ai-test"
STREAMLIT_PORT=8501
JUPYTER_PORT=8888

# Function to print colored output
print_info() {
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

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to cleanup previous test containers
cleanup() {
    print_info "Cleaning up previous test containers..."
    
    # Stop and remove container if it exists
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_info "Stopping and removing existing container: $CONTAINER_NAME"
        docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
        docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
    fi
    
    # Remove test image if it exists
    if docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^${DOCKER_IMAGE_NAME}:${DOCKER_TAG}$"; then
        print_info "Removing existing test image: ${DOCKER_IMAGE_NAME}:${DOCKER_TAG}"
        docker rmi "${DOCKER_IMAGE_NAME}:${DOCKER_TAG}" >/dev/null 2>&1 || true
    fi
}

# Function to validate environment
validate_environment() {
    print_info "Validating local environment..."
    
    # Check Docker
    if ! command_exists "docker"; then
        print_error "Docker not found. Please install Docker first."
        exit 1
    fi
    
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
    
    # Check for GPU support (optional)
    if docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
        print_success "Docker with GPU support detected"
        GPU_SUPPORT=true
    else
        print_warning "No GPU support detected. Running in CPU mode."
        GPU_SUPPORT=false
    fi
    
    # Check .env file
    if [[ ! -f ".env" ]]; then
        if [[ -f "env.template" ]]; then
            print_warning "Creating .env from template..."
            cp env.template .env
            print_info "Please edit .env file with your OpenAI API key"
            print_info "Minimum required: Set OPENAI_API_KEY in .env file"
            read -p "Press Enter after configuring .env file..."
        else
            print_error ".env file not found and no template available"
            exit 1
        fi
    fi
    
    # Check if OpenAI API key is set
    source .env 2>/dev/null || true
    if [[ -z "${OPENAI_API_KEY:-}" ]] || [[ "${OPENAI_API_KEY}" == "your_openai_api_key_here" ]]; then
        print_error "OPENAI_API_KEY not configured in .env file"
        print_info "Please edit .env and set OPENAI_API_KEY=your_actual_api_key"
        exit 1
    fi
    
    print_success "Environment validation completed"
}

# Function to build test Docker image
build_test_image() {
    print_info "Building test Docker image..."
    
    # Create a simplified Dockerfile for testing
    cat > Dockerfile.test << 'EOF'
# Simplified Dockerfile for local testing
FROM continuumio/miniconda3:latest

# Set environment variables
ENV PYTHONPATH=/workspace/src:$PYTHONPATH
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /workspace

# Copy simplified environment file and install dependencies
COPY environment-simple.yml ./environment.yml
RUN conda env create -f environment.yml

# Copy source code
COPY src/ ./src/
COPY .env .

# Create logs directory
RUN mkdir -p logs

# Create entrypoint script
RUN echo '#!/bin/bash\n\
source /opt/conda/etc/profile.d/conda.sh\n\
conda activate insulin-ai\n\
export PATH="/opt/conda/envs/insulin-ai/bin:$PATH"\n\
cd /workspace\n\
# Set up environment variables\n\
if [ -f .env ]; then\n\
    set -o allexport\n\
    source .env\n\
    set +o allexport\n\
fi\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["streamlit", "run", "src/insulin_ai/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
EOF
    
    # Build the test image
    if docker build -f Dockerfile.test -t "${DOCKER_IMAGE_NAME}:${DOCKER_TAG}" .; then
        print_success "Test Docker image built successfully"
    else
        print_error "Failed to build test Docker image"
        exit 1
    fi
    
    # Clean up test Dockerfile
    rm -f Dockerfile.test
}

# Function to test basic container functionality
test_container_basic() {
    print_info "Testing basic container functionality..."
    
    # Test if container can start and python works
    if docker run --rm "${DOCKER_IMAGE_NAME}:${DOCKER_TAG}" python --version; then
        print_success "Python is working in container"
    else
        print_error "Python test failed"
        return 1
    fi
    
    # Test if conda environment is activated
    if docker run --rm "${DOCKER_IMAGE_NAME}:${DOCKER_TAG}" conda info --envs | grep -q "insulin-ai"; then
        print_success "Conda environment 'insulin-ai' is available"
    else
        print_error "Conda environment test failed"
        return 1
    fi
    
    # Test if required packages are installed
    print_info "Testing required packages..."
    local packages=("streamlit" "openai" "langchain" "numpy" "pandas")
    
    for package in "${packages[@]}"; do
        if docker run --rm "${DOCKER_IMAGE_NAME}:${DOCKER_TAG}" python -c "import $package; print('$package OK')"; then
            print_success "$package is installed"
        else
            print_warning "$package import failed (might be optional)"
        fi
    done
}

# Function to start test container
start_test_container() {
    print_info "Starting test container..."
    
    local docker_args=""
    
    # Add GPU support if available
    if [[ "$GPU_SUPPORT" == true ]]; then
        docker_args="--gpus all"
    fi
    
    # Start container in detached mode
    if docker run -d \
        $docker_args \
        --name "$CONTAINER_NAME" \
        -p "${STREAMLIT_PORT}:8501" \
        -p "${JUPYTER_PORT}:8888" \
        --env-file .env \
        -v "$(pwd)/src:/workspace/src:rw" \
        -v "$(pwd)/data:/workspace/data:rw" \
        "${DOCKER_IMAGE_NAME}:${DOCKER_TAG}"; then
        print_success "Test container started successfully"
    else
        print_error "Failed to start test container"
        exit 1
    fi
}

# Function to wait for application to start
wait_for_application() {
    print_info "Waiting for Streamlit application to start..."
    
    local max_wait=120  # 2 minutes
    local wait_time=0
    local check_interval=5
    
    while [[ $wait_time -lt $max_wait ]]; do
        if curl -f http://localhost:${STREAMLIT_PORT} >/dev/null 2>&1; then
            print_success "Streamlit application is responding!"
            return 0
        fi
        
        print_info "Waiting... (${wait_time}s/${max_wait}s)"
        sleep $check_interval
        wait_time=$((wait_time + check_interval))
    done
    
    print_error "Application failed to start within ${max_wait} seconds"
    return 1
}

# Function to test application functionality
test_application() {
    print_info "Testing application functionality..."
    
    # Test if Streamlit is responding
    if curl -f http://localhost:${STREAMLIT_PORT} >/dev/null 2>&1; then
        print_success "Streamlit is responding on port ${STREAMLIT_PORT}"
    else
        print_error "Streamlit is not responding"
        return 1
    fi
    
    # Test if we can reach the main page
    if curl -s http://localhost:${STREAMLIT_PORT} | grep -q "Insulin"; then
        print_success "Application main page is accessible"
    else
        print_warning "Application main page content not detected (might be normal)"
    fi
    
    print_success "Application is running! You can access it at:"
    print_info "  Streamlit UI: http://localhost:${STREAMLIT_PORT}"
    print_info "  (Jupyter would be at: http://localhost:${JUPYTER_PORT})"
}

# Function to show container logs
show_logs() {
    print_info "Container logs (last 20 lines):"
    echo "----------------------------------------"
    docker logs --tail 20 "$CONTAINER_NAME" 2>&1 || true
    echo "----------------------------------------"
}

# Function to run interactive test
run_interactive_test() {
    print_info "Starting interactive test mode..."
    print_info "The application should be running at http://localhost:${STREAMLIT_PORT}"
    print_info ""
    print_info "Available commands:"
    print_info "  logs    - Show container logs"
    print_info "  status  - Show container status"
    print_info "  shell   - Open shell in container"
    print_info "  stop    - Stop the test"
    print_info "  help    - Show this help"
    print_info ""
    
    while true; do
        echo -n "Test> "
        read -r command
        
        case "$command" in
            "logs")
                show_logs
                ;;
            "status")
                docker ps --filter "name=$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
                ;;
            "shell")
                print_info "Opening shell in container (type 'exit' to return)..."
                docker exec -it "$CONTAINER_NAME" /bin/bash
                ;;
            "stop"|"quit"|"exit")
                print_info "Stopping interactive test..."
                break
                ;;
            "help"|"")
                print_info "Available commands: logs, status, shell, stop, help"
                ;;
            *)
                print_warning "Unknown command: $command (type 'help' for available commands)"
                ;;
        esac
    done
}

# Function to cleanup and exit
cleanup_and_exit() {
    print_info "Cleaning up test environment..."
    
    # Stop container
    if docker ps --filter "name=$CONTAINER_NAME" --quiet | grep -q .; then
        print_info "Stopping test container..."
        docker stop "$CONTAINER_NAME" >/dev/null 2>&1
    fi
    
    # Remove container
    if docker ps -a --filter "name=$CONTAINER_NAME" --quiet | grep -q .; then
        print_info "Removing test container..."
        docker rm "$CONTAINER_NAME" >/dev/null 2>&1
    fi
    
    print_success "Cleanup completed"
}

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Test the insulin-ai application locally using Docker"
    echo ""
    echo "Options:"
    echo "  --quick         Quick test (build and basic checks only)"
    echo "  --interactive   Interactive test mode (start app and allow manual testing)"
    echo "  --cleanup       Clean up previous test containers and exit"
    echo "  --help          Show this help message"
    echo ""
    echo "Default behavior: Build, test, and run interactive mode"
}

# Main function
main() {
    local quick_test=false
    local interactive_mode=true
    local cleanup_only=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --quick)
                quick_test=true
                interactive_mode=false
                shift
                ;;
            --interactive)
                interactive_mode=true
                shift
                ;;
            --cleanup)
                cleanup_only=true
                shift
                ;;
            --help)
                usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    # Handle cleanup only
    if [[ "$cleanup_only" == true ]]; then
        cleanup
        exit 0
    fi
    
    # Set up trap for cleanup on exit
    trap cleanup_and_exit EXIT INT TERM
    
    print_info "Starting local testing for Insulin-AI Docker setup"
    print_info "======================================================="
    
    # Run tests
    validate_environment
    cleanup
    build_test_image
    test_container_basic
    
    if [[ "$quick_test" == true ]]; then
        print_success "Quick test completed successfully!"
        print_info "The Docker setup appears to be working correctly."
        exit 0
    fi
    
    # Full test with application
    start_test_container
    
    if wait_for_application; then
        test_application
        
        if [[ "$interactive_mode" == true ]]; then
            run_interactive_test
        else
            print_success "Application test completed successfully!"
            sleep 5  # Give user time to see the success message
        fi
    else
        print_error "Application failed to start properly"
        show_logs
        exit 1
    fi
    
    print_success "Local testing completed successfully!"
    print_info "Your Docker setup is ready for vast.ai deployment."
}

# Run main function with all arguments
main "$@" 