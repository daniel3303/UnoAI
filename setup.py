from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="uno-ai",
    version="0.1.0",
    author="Daniel Oliveira",
    author_email="hello@danielapoliveira.com",
    description="UNO AI using Reinforcement Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/daniel3303/UnoAI",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.12",
    install_requires=requirements,
    extras_require={
        "training": [
            "wandb>=0.12.0",
            "tensorboard>=2.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # Original training scripts
            "uno-train-ppo=uno_ai.training.train_ppo:main",
            "uno-train-supervised=uno_ai.training.trainer:main",

            # Multi-agent training scripts
            "uno-train-multi-agent=uno_ai.training.train_multi_agent_ppo:main",
            "uno-train-multi-agent-parallel=uno_ai.training.train_parallel_multi_agent_ppo:main",

            # Evaluation and demo scripts
            "uno-evaluate=uno_ai.training.evaluate:main",
            "uno-demo=uno_ai.training.demo:main",
        ],
    },
    package_data={
        'uno_ai': ['assets/images/cards/**/*.jpg'],
    },
    include_package_data=True,
    zip_safe=False,
)