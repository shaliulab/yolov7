from setuptools import setup, find_packages

setup(
    name='yolov7',
    version='0.1.0',  # Your package version
    author='Antonio Ortega',
    author_email='antonio.ortega@kuleuven.be',
    description='Package for YOLOv7',
    # long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    # url='https://github.com/shaliulab/behavior-viewer',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved',
        'Operating System :: OS Independent',
    ],
    # python_requires='>=3.10.11',  # Specify your minimal Python version
    install_requires=[
        "pillow", "torch", "torchvision",
    #     "pandas>=1.3.5",
    #     "plotly>=5.16.1",
    #     "joblib>=1.3.2",
    #     # Add more packages as needed
    ],
    entry_points={
        'console_scripts': [
            'yolov7-detect=yolov7.detect:main',
        ],
    },

)
