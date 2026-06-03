# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /

# Install dependencies
RUN pip install rpyc numpy xarray scipy netCDF4 rasterio h5py matplotlib statsmodels scikit-learn

# Other setup for your container if needed
COPY docker/utils/python_server.py /

# Run the server
CMD ["python", "python_server.py"]
