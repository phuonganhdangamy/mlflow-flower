#!/bin/bash

set -e  # Exit on any error

# Configuration
NUM_CLIENTS=5
NUM_ROUNDS=5
SERVER_PORT=8080
MODULE_FILE="taxi_utils_native_keras.py"
OUTPUT_DIR="./tfx_output"
DATA_ROOT_DIR="../tfx-flower/data/simple"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_section() {
    echo -e "\n${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if a port i available
check_port() {
    if command_exists netstat; then
        if netstat -tuln | grep -q ":$1 "; then
            return 1
        fi
    elif command_exists ss; then
        if ss -tuln | grep -q ":$1 "; then
            return 1
        fi
    fi
    return 0
}

# Wait for the server to be ready
wait_for_server() {
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if command_exists nc; then
            if nc -z localhost $SERVER_PORT 2>/dev/null; then
                return 0
            fi
        elif command_exists telnet; then
            if timeout 1 telnet localhost $SERVER_PORT >/dev/null 2>&1; then
                return 0
            fi
        else
            # Fallback: just wait a fixed time
            if [ $attempt -eq 5 ]; then
                return 0
            fi
        fi
        
        echo "Waiting for server to start... (attempt $attempt/$max_attempts)"
        sleep 2
        attempt=$((attempt + 1))
    done
    
    return 1
}

# Clean up background processes
cleanup() {
    print_section "CLEANING UP PROCESSES"
    if [ ! -z "$SERVER_PID" ] && kill -0 $SERVER_PID 2>/dev/null; then
        print_warning "Stopping server (PID: $SERVER_PID)"
        kill $SERVER_PID
        wait $SERVER_PID 2>/dev/null || true
    fi
    
    for pid in "${CLIENT_PIDS[@]}"; do
        if [ ! -z "$pid" ] && kill -0 $pid 2>/dev/null; then
            print_warning "Stopping client (PID: $pid)"
            kill $pid
            wait $pid 2>/dev/null || true
        fi
    done
    
    print_success "Cleanup completed"
}

# Set up cleanup trap
trap cleanup EXIT INT TERM

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-clients)
            NUM_CLIENTS="$2"
            shift 2
            ;;
        --num-rounds)
            NUM_ROUNDS="$2"
            shift 2
            ;;
        --port)
            SERVER_PORT="$2"
            shift 2
            ;;
        --module-file)
            MODULE_FILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --data-root-dir)
            DATA_ROOT_DIR="$2"
            shift 2
            ;;
        --skip-pipelines)
            SKIP_PIPELINES=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --num-clients NUM      Number of clients (default: 5)"
            echo "  --num-rounds NUM       Number of federated rounds (default: 5)"
            echo "  --port PORT           Server port (default: 8080)"
            echo "  --module-file FILE    TFX module file (default: taxi_utils_native_keras.py)"
            echo "  --output-dir DIR      TFX output directory (default: ./tfx_output)"
            echo "  --data-root-dir DIR   Data root directory (default: ../tfx-flower/data/simple)"
            echo "  --skip-pipelines      Skip TFX pipeline execution"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

print_section "FEDERATED LEARNING SETUP"
echo "Configuration:"
echo "  - Number of clients: $NUM_CLIENTS"
echo "  - Number of rounds: $NUM_ROUNDS"
echo "  - Server port: $SERVER_PORT"
echo "  - Module file: $MODULE_FILE"
echo "  - Output directory: $OUTPUT_DIR"
echo "  - Data root directory: $DATA_ROOT_DIR"

# Check prerequisites
print_section "CHECKING PREREQUISITES"

if ! command_exists python; then
    print_error "Python not found. Please install Python."
    exit 1
fi
print_success "Python found"

if ! python -c "import flwr" 2>/dev/null; then
    print_error "Flower not installed. Run: pip install -r requirements.txt"
    exit 1
fi
print_success "Flower installed"

if ! python -c "import tfx" 2>/dev/null; then
    print_error "TFX not installed. Run: pip install -r requirements.txt"
    exit 1
fi
print_success "TFX installed"

if [ ! -f "$MODULE_FILE" ]; then
    print_error "Module file not found: $MODULE_FILE"
    print_warning "Please create the TFX preprocessing module file"
    exit 1
fi
print_success "Module file found: $MODULE_FILE"

if ! check_port $SERVER_PORT; then
    print_error "Port $SERVER_PORT is already in use"
    exit 1
fi
print_success "Port $SERVER_PORT is available"


# Start Federated Learning server
print_section "STARTING FEDERATED LEARNING SERVER"

python server.py

SERVER_PID=$!
print_success "Server started (PID: $SERVER_PID)"

# Wait for server to be ready
if ! wait_for_server; then
    print_error "Server failed to start or is not responding"
    exit 1
fi
print_success "Server is ready and accepting connections"


# Start clients
print_section "STARTING FEDERATED LEARNING CLIENTS"
CLIENT_PIDS=()
for ((i=0; i<NUM_CLIENTS; i++)); do
    echo "Starting client $i..."
    python client.py \
        --client-id "$i" \
        --server-address "localhost:$SERVER_PORT" \
        --output-dir "$OUTPUT_DIR" &
    
    client_pid=$!
    CLIENT_PIDS+=($client_pid)
    print_success "Client $i started (PID: $client_pid)"
    
    # Small delay between client starts
    sleep 1
done

# Wait for completion
print_section "FEDERATED LEARNING IN PROGRESS"
# Wait for server to complete
if wait $SERVER_PID; then
    print_success "Federated learning completed successfully!"
else
    print_error "Server process ended unexpectedly"
    exit 1
fi

print_section "FEDERATED LEARNING COMPLETED"
print_success "All processes completed successfully"
