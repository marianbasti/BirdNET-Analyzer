<div align="center">
  <h1>BirdNET-Analyzer</h1>
    <a href="https://birdnet-team.github.io/BirdNET-Analyzer/">
        <img src="https://github.com/birdnet-team/BirdNET-Analyzer/blob/main/docs/_static/logo_birdnet_big.png?raw=true" width="300" alt="BirdNET-Logo" />
    </a>
</div>
<br>
<div align="center">

![Licencia](https://img.shields.io/github/license/birdnet-team/BirdNET-Analyzer)
![SO](https://badgen.net/badge/OS/Linux%2C%20Windows%2C%20macOS/blue)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
![Especies](https://badgen.net/badge/Species/6512/blue)
![Descargas](https://www-user.tu-chemnitz.de/~johau/birdnet_total_downloads_badge.php)

[![Docker](https://github.com/birdnet-team/BirdNET-Analyzer/actions/workflows/docker-build.yml/badge.svg)](https://github.com/birdnet-team/BirdNET-Analyzer/actions/workflows/docker-build.yml)
[![Reddit](https://img.shields.io/badge/Reddit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/BirdNET_Analyzer/)
![Estrellas de GitHub)](https://img.shields.io/github/stars/birdnet-team/BirdNET-Analyzer)

[![GitHub release](https://img.shields.io/github/v/release/birdnet-team/BirdNET-Analyzer)](https://github.com/birdnet-team/BirdNET-Analyzer/releases/latest)
[![PyPI - Versión](https://img.shields.io/pypi/v/birdnet_analyzer?logo=pypi)](https://pypi.org/project/birdnet-analyzer/)

[![Patrocinar](https://img.shields.io/badge/Support%20our%20work-8A2BE2?logo=data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjE2IiB2aWV3Qm94PSIwIDAgMTYgMTYiIHZlcnNpb249IjEuMSIgd2lkdGg9IjE2IiBkYXRhLXZpZXctY29tcG9uZW50PSJ0cnVlIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPg0KICAgIDxwYXRoIGQ9Im04IDE0LjI1LjM0NS42NjZhLjc1Ljc1IDAgMCAxLS42OSAwbC0uMDA4LS4wMDQtLjAxOC0uMDFhNy4xNTIgNy4xNTIgMCAwIDEtLjMxLS4xNyAyMi4wNTUgMjIuMDU1IDAgMCAxLTMuNDM0LTIuNDE0QzIuMDQ1IDEwLjczMSAwIDguMzUgMCA1LjUgMCAyLjgzNiAyLjA4NiAxIDQuMjUgMSA1Ljc5NyAxIDcuMTUzIDEuODAyIDggMy4wMiA4Ljg0NyAxLjgwMiAxMC4yMDMgMSAxMS43NSAxIDEzLjkxNCAxIDE2IDIuODM2IDE2IDUuNWMwIDIuODUtMi4wNDUgNS4yMzEtMy44ODUgNi44MThhMjIuMDY2IDIyLjA2NiAwIDAgMS0zLjc0NCAyLjU4NGwtLjAxOC4wMS0uMDA2LjAwM2gtLjAwMlpNNC4yNSAyLjVjLTEuMzM2IDAtMi43NSAxLjE2NC0yLjc1IDMgMCAyLjE1IDEuNTggNC4xNDQgMy4zNjUgNS42ODJBMjAuNTggMjAuNTggMCAwIDAgOCAxMy4zOTNhMjAuNTggMjAuNTggMCAwIDAgMy4xMzUtMi4yMTFDMTIuOTIgOS42NDQgMTQuNSA3LjY1IDE0LjUgNS41YzAtMS44MzYtMS40MTQtMy0yLjc1LTMtMS4zNzMgMC0yLjYwOS45ODYtMy4wMjkgMi40NTZhLjc0OS43NDkgMCAwIDEtMS40NDIgMEM2Ljg1OSAzLjQ4NiA1LjYyMyAyLjUgNC4yNSAyLjVaIj48L3BhdGg+DQo8L3N2Zz4=)](https://give.birds.cornell.edu/page/132162/donate/1)

</div>

Este repositorio contiene scripts de BirdNET para procesar grandes cantidades de datos de audio o archivos de audio individuales.
Esta es la versión más avanzada de BirdNET para análisis acústicos y mantendremos este repositorio actualizado con nuevos modelos e interfaces mejoradas para permitir que científicos sin experiencia en informática puedan ejecutar el análisis.

Siéntete libre de usar BirdNET para tus análisis acústicos e investigaciones.
Si lo haces, por favor cita como:

```bibtex
@article{kahl2021birdnet,
  title={BirdNET: A deep learning solution for avian diversity monitoring},
  author={Kahl, Stefan and Wood, Connor M and Eibl, Maximilian and Klinck, Holger},
  journal={Ecological Informatics},
  volume={61},
  pages={101236},
  year={2021},
  publisher={Elsevier}
}
```

## Documentación

Puedes acceder a la documentación de este proyecto [aquí](https://birdnet-team.github.io/BirdNET-Analyzer/).


## Implementación en Pytorch

El modelo ha sido reimplementado en PyTorch. Para usarlo, puedes hacerlo instalando los módulos requeridos y ejecutando los siguientes comandos:

### Instalando dependencias
```bash
python3 -m venv .venv
source .venv/bin/activate  # En Windows usa: .venv\Scripts\activate
pip install -r requirements.txt
``` 

### Ejecutando la interfaz web
```bash
python -m birdnet_analyzer.torch_gradio
```

Esto iniciará una interfaz web en tu máquina local, a la que puedes acceder vía `http://localhost:7860`.

## Descarga

Puedes descargar instaladores para Windows y macOS desde la [página de lanzamientos](https://github.com/birdnet-team/BirdNET-Analyzer/releases/latest).
Los modelos se pueden encontrar en [Zenodo](https://zenodo.org/records/15050749).

## Acerca de

Desarrollado por el [K. Lisa Yang Center for Conservation Bioacoustics](https://www.birds.cornell.edu/ccb/) en el [Cornell Lab of Ornithology](https://www.birds.cornell.edu/home) en colaboración con la [Universidad Técnica de Chemnitz](https://www.tu-chemnitz.de/index.html.en).

Visita https://birdnet.cornell.edu para saber más sobre el proyecto.

¿Quieres usar BirdNET para analizar un gran conjunto de datos? No dudes en contactarnos: ccb-birdnet@cornell.edu

**¿Tienes una pregunta, comentario o solicitud de función? Por favor, inicia un nuevo hilo de issue para hacérnoslo saber. Siéntete libre de enviar un pull request.**

## Licencia

- **Código fuente**: El código fuente de este proyecto está licenciado bajo la [Licencia MIT](https://opensource.org/licenses/MIT).
- **Modelos**: Los modelos usados en este proyecto están licenciados bajo la [Licencia Creative Commons Atribución-NoComercial-CompartirIgual 4.0 Internacional (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

Por favor, asegúrate de revisar y cumplir con los términos específicos de la licencia proporcionados con cada modelo.

*Ten en cuenta que todos los propósitos educativos y de investigación se consideran uso no comercial y, por lo tanto, está permitido usar los modelos BirdNET de cualquier manera.*

## Financiación

Este proyecto es apoyado por Jake Holshuh (Cornell clase del ´69) y The Arthur Vining Davis Foundations.
Nuestro trabajo en el K. Lisa Yang Center for Conservation Bioacoustics es posible gracias a la generosidad de K. Lisa Yang para avanzar en tecnologías innovadoras de conservación que inspiren e informen la conservación de la vida silvestre y los hábitats.

El desarrollo de BirdNET está apoyado por el Ministerio Federal Alemán de Educación e Investigación a través del proyecto “BirdNET+” (FKZ 01|S22072).
El Ministerio Federal Alemán de Medio Ambiente, Conservación de la Naturaleza y Seguridad Nuclear contribuye a través del proyecto “DeepBirdDetect” (FKZ 67KI31040E).
Además, la Deutsche Bundesstiftung Umwelt apoya BirdNET a través del proyecto “RangerSound” (proyecto 39263/01).

## Socios

BirdNET es un esfuerzo conjunto de socios de la academia y la industria.
Sin estas alianzas, este proyecto no habría sido posible.
¡Gracias!

![Logos de todos los socios](https://tuc.cloud/index.php/s/KSdWfX5CnSRpRgQ/download/box_logos.png)
