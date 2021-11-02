import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="grokking",
    version="0.0.1",
    author="Martin Weiss",
    author_email="martin.clyde.weiss@gmail.com",
    description="Reproducing OpenAI's Grokking paper",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[
        'torch==1.9.1',
        'numpy',
        'tfds-nightly',
        't5',
        'seqio',
        'wandb',
        'google-cloud-storage',
        'speedrun @ git+https://git@github.com/inferno-pytorch/speedrun@dev#egg=speedrun',
        'wormulon @ git+https://git@github.com/mweiss17/wormulon/@main#egg=wormulon'
        ],
    extras_require={
        "xla": ['torch-xla @ https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-1.9-cp38-cp38-linux_x86_64.whl'],
    }
)

