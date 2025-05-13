
# Download GAIA dataset
if [ ! -d "examples/gaia/GAIA" ]; then
    git clone git@hf.co:datasets/gaia-benchmark/GAIA examples/gaia/GAIA
fi

# Build docker image
docker compose up -d