# 🔬 Analisador de Corrosão em Chapas Metálicas (Visão Computacional)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Scientific-013243?logo=numpy&logoColor=white)

**Software desktop de visão computacional em Python para detecção, segmentação e quantificação automática de áreas corroídas em chapas de alumínio e metais.**

O **Analisador de Corrosão** foi desenvolvido no contexto de pesquisa científica (PIBIC) para substituir inspeções visuais subjetivas por medições geométricas quantitativas e reprodutíveis. Utilizando processamento digital de imagens no espaço de cor **CIE LAB**, operações morfológicas e estatística robusta, o software isola a região útil da chapa, ignora furos/marcações de fixação, segmenta regiões oxidadas e gera relatórios completos com classificação aproximada segundo a norma **ASTM D610**.

---

## 🎯 Objetivos do Projeto

* Automatizar a segmentação de corrosão em fotografias de ensaios laboratoriais (JPG, PNG, BMP, TIFF, WEBP) com processamento em lote.
* Utilizar o espaço de cor **CIE LAB** para separar luminância de cromaticidade, minimizando interferências de reflexos metálicos e iluminação.
* Calcular métricas geométricas precisas (área total, percentual de corrosão, contagem de pites e conversão física em $\text{mm}^2$ via calibração `px/mm`).
* Exportar imagens anotadas (*overlays* visuais) e relatórios estruturados em **CSV** e **JSON** sem travar a interface gráfica (*multithreading*).

---

## 🛠️ Tecnologias Utilizadas

* **Python 3.x**
* **OpenCV (`cv2`)** – Conversão de espaço de cor (CIE LAB), segmentação, operações morfológicas e geração de *overlays*
* **NumPy** – Operações matriciais de alta performance e estatística robusta
* **Tkinter & Threading/Queue** – Interface gráfica desktop responsiva com fila de processamento assíncrono em lote
* **CSV & JSON** – Exportação automatizada de métricas laboratoriais

---

## 🚀 Como Executar o Projeto

### Pré-requisitos
Certifique-se de ter o **Python 3.8+** instalado em sua máquina.

### Passo a passo

1. **Clone este repositório:**
   ```bash
   git clone https://github.com/vitorjazz-lang/Analisador-de-Corrosao.git
   cd analisador-corrosao-opencv
   ```

2. **Crie e ative um ambiente virtual (opcional, mas recomendado):**
   ```bash
   python -m venv venv
   # No Windows:
   venv\Scripts\activate
   # No Linux/Mac:
   source venv/bin/activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Execute a aplicação:**
   ```bash
   python analisador_corrosao.py
   ```
   *(Selecione as fotografias da chapa, ajuste a sensibilidade se necessário e confira os resultados gerados na pasta `resultados_corrosao/`)*.

---

## 🤝 Contribuição

Contribuições acadêmicas ou técnicas (como novos filtros de iluminação, calibração automática por régua ou classificadores auxiliares) são bem-vindas! Abra uma *Issue* ou envie um *Pull Request*.
