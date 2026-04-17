FROM ubuntu:26.04
# Install C compiler, system Python and downloaders
RUN apt-get update && apt-get install -y gcc build-essential python3 python3-dev curl
# Install & set up uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
# Create Python virtual environment
RUN uv venv /opt/jlcdm
# Activate for subsequent RUN/CMD steps
ENV VIRTUAL_ENV=/opt/jlcdm
ENV PATH="/opt/jlcdm/bin:$PATH"
# Activate for interactive shells
RUN echo 'source /opt/jlcdm/bin/activate' >> ~/.bashrc
