
# Check your env file
if [ ! -f ".env" ]; then
    echo "Please add your own .env config file from template .env.template before running gaia test!"
    exit 1
fi

# Build docker image
docker compose up --build -d