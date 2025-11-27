FROM julia:1.11
RUN apt-get update && \
    apt-get install -y ca-certificates && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*
# DEBUG ONLY: Tell Julia to trust everyone (Dangerous)
ENV JULIA_SSL_NO_VERIFY_HOSTS="**"

WORKDIR /app

COPY mcmc/Project.toml Project.toml
COPY data/ ./data/
RUN julia --project=. -e 'using Pkg; Pkg.add(url="https://github.com/patrickm663/hmd.jl")'
RUN julia --project=. -e 'using Pkg; Pkg.instantiate();'
RUN julia --project=. -e 'using Pkg; Pkg.precompile();'
RUN julia --project=. -e 'using Pkg; Pkg.rm("Turing"); Pkg.add(name="Turing", version="v0.34.1");'

EXPOSE 1234

COPY index.jl index.jl

# launch_browser=false prevents it from trying to open a browser inside the server
CMD ["julia", "--project=.", "-e", "import Pluto; Pluto.run(notebook=\"index.jl\", host=\"0.0.0.0\", port=1234, launch_browser=false)"]
