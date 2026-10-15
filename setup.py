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
        "pillow", "torch", "torchvision", "pyyaml",
    ],
    extras_require={
        # yolov7-data; reading FlyHostel stores in detect.py additionally needs flyhostel
        "workflow": ["fiftyone", "cvat-sdk"],
    },
    entry_points={
        'console_scripts': [
            'yolov7-detect=yolov7.detect:main',
            'yolov7-data=yolov7.workflow.data:main',
            'yolov7-train=yolov7.workflow.train:main',
            'yolov7-models=yolov7.workflow.models:main',
        ],
    },

)
