from setuptools import setup, find_packages

setup(
    name="ner-tools",  # Replace with your actual project name
    version="0.1.0",
    
    # Automatically discover and include all packages
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    
    # Metadata about your project
    author="Arthur Weber Ziegler",
    author_email="arthur.ziegler@gmail.com",
    description="This project uses NER to classify entities in documents from UFRGS.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
        
    # Specify Python versions supported
    python_requires='>=3.8',
    
)