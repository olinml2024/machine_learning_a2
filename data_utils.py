import gdown

def download_dessert_test_data():
    gdown.download(id="1k_KaPCCO37HC8eqpd0D0JhKyGhqM8_KL", output="test_data.zip")
    !mkdir test_data
    !unzip -q test_data.zip -d test_data
