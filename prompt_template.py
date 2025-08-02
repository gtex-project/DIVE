from typing import List, Dict, Any
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

def get_text_response_schema() -> List[ResponseSchema]:
    return [
        ResponseSchema(name="Formula", description="Chemical formula of the material or the same formula prepared under different conditions"),
        ResponseSchema(
            name="Material type",
            description=(
                "Specify the type of hydrogen storage material. Choose from the following options:\n"
                "- Interstitial Hydride (Hydrogen atoms occupy interstitial sites in metal lattices，e.g., TiFe, LaNi₅, ZrV₂, TiMn₂, ZrNi, FeTi, Mg₂Ni, VTiCr, ScAlMn, TiCrMn)\n"
                "- Complex Hydride (Contain covalently bonded anionic complexes such as [AlH₄]⁻, [BH₄]⁻, etc. e.g., NaAlH₄, LiAlH₄, LiBH₄, Mg(BH₄)₂, Ca(BH₄)₂, LiNH₂, Mg(NH₂)₂, NH₃BH₃ (ammonia borane), KAlH₄, NaZn(BH₄)₃, Li₄BN₃H₁₀, Na₂LiAlH₆)\n"
                "- Ionic Hydrides (Formed by electropositive metals and H⁻ ions, typically saline-like in structure. e.g., LiH, NaH, KH, RbH, CsH, CaH₂, SrH₂, BaH₂, MgH₂, BeH₂)\n"
                "- Porous materials (Hydrogen is stored by physisorption or chemisorption within high surface area frameworks. e.g., COFs (e.g., COF-1, COF-102), MOFs (e.g., MOF-5, UiO-66, HKUST-1, ZIF-8), porous carbon, activated carbon, graphene, carbon nanotubes (CNTs), zeolites)\n"
                "- Multi-component Hydrides (Combinations or composites of different hydride types to improve storage properties. e.g., mixtures of interstitial, ionic or complex hydrides)\n"
                "- Superhydrides (Hydrides with exceptionally high hydrogen content (typically x > 5 in MHₓ), often formed under high  (GPa). e.g. LaH10, YH9, CaH6, ThH10, ScH9, Y3Fe4H20, UH8, CeH9, AcH10, BaH12) Materials with large hydrogen content or containing hydrogen-rich complex anions (e.g., FeH₈ units) may also be referred to as superhydrides, even if they are formally categorized as complex hydrides—such as Y₃Fe₄H₂₀.\n"
                "- Others (Materials that do not clearly fit into the above categories)"
            )
        ),
        ResponseSchema(
            name="Interstitial Hydride Category",
            description=(
                "If the material type is 'Interstitial Hydride', specify its subcategory. Select one of the following (or a similar category):\n"
                "- AB2: Composed of A-site and B-site metal elements with an overall atomic ratio of approximately 1:2 (A:B). "
                "Either A or B can include multiple elements, as long as the total atomic ratio meets the AB2 requirement. "
                "Typical structures include Laves phases (e.g., TiMn2, (Ti,Zr)(Mn,Cr)2).\n"
                "- AB3: Contains A-site and B-site elements with an overall atomic ratio of about 1:3 (A:B). "
                "A or B can be multi-elemental, provided the total ratio is AB3. "
                "The structure combines AB2 (Laves polyhedron) and AB5 (capped hexagonal prism) units (e.g., LaNi3).\n"
                "- AB5: Made up of A-site and B-site metal elements with a total atomic ratio of about 1:5 (A:B). "
                "Multiple elements are allowed in A or B sites as long as the AB5 stoichiometry is satisfied. "
                "Mainly consists of capped hexagonal prism units (e.g., LaNi5, (La,Ce)(Ni,Co,Al)5).\n"
                "- Others"
            )
        ),
    ]

def get_material_type_template() -> PromptTemplate:
    text_response_schema = get_text_response_schema()
    text_parser = StructuredOutputParser.from_response_schemas(text_response_schema)
    return PromptTemplate(
        template="""
Below is the text from a scientific publication on hydrogen storage materials.  
First, identify and extract the chemical formula of the newly reported material in the article. Next, determine the material’s main type and sub-type based on the provided classification system.

Paper Text:
{text}

Respond in JSON format following this schema: {parser_schema}
/no_think 
    """,
        input_variables=["text"],
        partial_variables={"parser_schema": text_parser.get_format_instructions()}
    )

def get_reformat_prompt() -> PromptTemplate:
    text_response_schema = [
        ResponseSchema(name="ID", description="String. Sequential integer like '1', '2', etc."),
        ResponseSchema(name="doi", description="String. DOI in the format '10.xxxx/yyyy'. No URL prefix."),
        ResponseSchema(name="Formula", description="String. Chemical formula."),
        ResponseSchema(name="Metal Alloy Category", description="String. One of: 'AB2', 'AB3', 'AB5', 'Others'."),
        ResponseSchema(name="Hydrogenation temperature", description="List of length 2: ['value', 'unit']. Unit must be 'K' or '°C'."),
        ResponseSchema(name="Hydrogenation pressure", description="List of length 2: ['value', 'unit']. Unit must be 'bar' or 'MPa'."),
        ResponseSchema(name="Dehydrogenation temperature", description="List of length 2: ['value', 'unit']. Unit must be 'K' or '°C'."),
        ResponseSchema(name="Dehydrogenation pressure", description="List of length 2: ['value', 'unit']. Unit must be 'bar' or 'MPa'."),
        ResponseSchema(name="Volumetric hydrogen capacity", description="List of length 6: ['value', 'unit', 'pressure', 'pressure unit', 'temperature', 'temperature unit']."),
        ResponseSchema(name="Gravimetric hydrogen density", description="List of length 6: ['value', 'unit', 'pressure', 'pressure unit', 'temperature', 'temperature unit']."),
        ResponseSchema(name="H2 experiment release", description="List of length 2: ['value', 'unit']."),
        ResponseSchema(name="H2 experiment adsorption", description="List of length 2: ['value', 'unit']."),
        ResponseSchema(name="enthalpy change (ΔH, kJ/mol H2)", description="List of length 4: ['hydrogenation value', 'unit', 'dehydrogenation value', 'unit']."),
        ResponseSchema(name="entropy change (ΔS, J/mol H2·K)", description="List of length 4: ['hydrogenation value', 'unit', 'dehydrogenation value', 'unit']."),
        ResponseSchema(name="Material type", description="String. One of: 'Metal Hydrides', 'Metal Alloy', 'MOFs', 'COFs', 'Porous Carbon', 'Others'."),
        ResponseSchema(name="Publication year", description="String. Four-digit year (e.g., '2013').")
    ]
    text_parser = StructuredOutputParser.from_response_schemas(text_response_schema)
    return PromptTemplate(
        template="""
You are given a list of dictionaries extracted from scientific literature and a summary of information obtained from diagrams.

List of dictionaries:
{text}

Your task is to reformat the input into a new list of dictionaries in JSON format.
Each dictionary must strictly conform to the following schema:
{parser_schema}

Instructions:
- Return the result as a valid JSON-formatted **list of dictionaries**.
- Ensure that **each dictionary contains all required fields** with values formatted exactly as described in the schema.
- For fields that are specified as lists, ensure the **list length matches the required format** (e.g., length 2, 4, or 6).
- Normalize all empty or missing values to the string **"NA"**. This includes:
    - null values: `null`, `None`, `NaN`, `nan`
    - empty strings: ""
    - any other placeholders indicating missing data

Do not include any explanation—output only the JSON list.
/no_think
""",
        input_variables=["text"],
        partial_variables={"parser_schema": text_parser.get_format_instructions()}
    )

def get_image_pct_prompt() -> PromptTemplate:
    image_response_schema = [
        ResponseSchema(
            name="ID", 
            description="Sequential integer number (e.g., '1', '2'...). This integer corresponds to the order of extracted curves from the image."
        ),
        ResponseSchema(
            name="Formula", 
            description="Chemical formula of the material, including distinguishing features if the same formula appears under different preparation conditions (e.g., 'MgH₂ (ball-milled)', 'Ti-Fe alloy (annealed at 500°C)', 'Ni-MH (as-cast)'). Use standard element symbols and numeric subscripts where applicable."
        ),
        ResponseSchema(
            name="Description of the hydrogenation process", 
            description="Description of hydrogen absorption process (fill the variables in {}).  Output format: The x- and y-axis units are {[x-axis unit, y-axis unit]}. At {temperature}, absorption starts at {[x0, y0]}, reaches one or more pressure plateau {p0, p1, ...}. The hydrogen storage density after the final pressure plateau is {w0}. Finally, the maximum hydrogen storage density is at {[x2, y2]}."
        ),
        ResponseSchema(
            name="Description of the dehydrogenation process", 
            description="Description of hydrogen desorption process (fill the variables in {}). Output format: The x- and y-axis units are {[x-axis unit, y-axis unit]}. At {temperature}, desorption starts at {[x2, y2]}, reaches one or more pressure plateau {p0, p1, ...}. The hydrogen storage density at the start of the first pressure plateau during the dehydrogenation process is {w1}. Finally, the minimum hydrogen storage density is at {[x3, y3]}."
        ),
        ResponseSchema(
            name="Gravimetric hydrogen density", 
            description="Maximum hydrogen storage capacity for each curve, along with the corresponding pressure and temperature. this value is {x2}. Output format: ['Value', 'unit', 'pressure value', 'pressure unit', 'temperature value', 'temperature unit']."
        ),
        ResponseSchema(
            name="Gravimetric hydrogen density at 0.01 MPa", 
            description="Hydrogen storage capacity at 0.01 MPa for each curve, with temperature. If unavailable, record value as 'NA'. Output format: ['Value', 'unit', '0.01', 'MPa', 'temperature value', 'temperature unit']."
        ),
        ResponseSchema(
            name="Gravimetric hydrogen density at 0.1 MPa", 
            description="Hydrogen storage capacity at 0.1 MPa for each curve, with temperature. If unavailable, record value as 'NA'. Output format: ['Value', 'unit', '0.1', 'MPa', 'temperature value', 'temperature unit']."
        ),
        ResponseSchema(
            name="Gravimetric hydrogen density at 1 MPa", 
            description="Hydrogen storage capacity at 1 MPa for each curve, with temperature. If unavailable, record value as 'NA'. Output format: ['Value', 'unit', '1', 'MPa', 'temperature value', 'temperature unit']."
        ),
        ResponseSchema(
            name="Gravimetric hydrogen density at 10 MPa", 
            description="Hydrogen storage capacity at 10 MPa for each curve, with temperature. If unavailable, record value as 'NA'. Output format: ['Value', 'unit', '10', 'MPa', 'temperature value', 'temperature unit']."
        ),
        ResponseSchema(
            name="H2 experiment release", 
            description="Based on the description of the dehydrogenation process, this value is {w1} - {x3}. Output format: ['value', 'unit']."
        ),
        ResponseSchema(
            name="H2 experiment adsorption", 
            description="Based on the description of the hydrogenation process, this value is {w0} - {x0}. Output format: ['value', 'unit']."
        ),
    ]
    image_parser = StructuredOutputParser.from_response_schemas(image_response_schema)
    return PromptTemplate(
        template="""       
Please extract information from the following figures step by step using the provided context. 

Figure Context:
{caption_context}

For each figure, follow the exact instructions below:
Step 1: Figure Identification and Contextual Summary
Identify which figure the image corresponds to from the Figure Context.
Summarize the relevant figure context in plain text, including the material formula, temperature, testing condition, and any keywords such as "reversible", "equilibrium pressure", "plateau", etc.
Format: a natural paragraph. Do not output JSON here.

Step 2: Curve Extraction and Grouping
If the figure contains multiple subplots, treat each subplot independently.
In each subplot, identify all absorption and desorption cycle curves. If possible, use legends, colors, or arrow directions to distinguish absorption from desorption.

For each complete hydrogenation–dehydrogenation cycle (if only hydrogenation or dehydrogenation is present, record the available data), extract the information as a single dictionary. Group the corresponding absorption and desorption data together.
For each such cycle, create one dictionary following the specified JSON schema: 
{parser_schema}

Important:
Define the equilibrium pressure as the average pressure in the central region of the pressure plateau observed during hydrogen absorption or desorption in the PCT measurement.
If the y-axis of the PCT plot is shown on a logarithmic scale, first convert it to a linear scale before identifying the pressure plateau.
    
The final output should be a JSON-formatted list (array) of these dictionaries, with each dictionary representing one full absorption-desorption cycle under given conditions.
Output strictly in valid JSON format.


        """,
        input_variables=["caption_context"],
        partial_variables={"parser_schema": image_parser.get_format_instructions()}
    )

def get_image_discharge_prompt() -> PromptTemplate:
    image_response_schema = [
        ResponseSchema(name="ID", description="Sequential number (e.g., '1', '2'), this integer corresponds to the order of the extracted Python dictionary."),
        ResponseSchema(name="Formula", description="Chemical formula of the material or of the material prepared under different conditions. It must be a combination of element symbols and numbers."),
        ResponseSchema(name="Discharge Ability", description="Initial discharge ability of the formula material, output format: ['Value', 'unit (e.g., mAh g^-1)']"),
        ResponseSchema(name="Discharge Ability After Cycles", description="Final discharge ability of the formula material after cycle measurements, output format: ['Value', 'unit (e.g., mAh g^-1)', 'cycle number']"),    
        ResponseSchema(name="Gravimetric hydrogen density", description="Convert the initial discharge ability into gravimetric hydrogen storage percentage based on the relation:1 wt% H₂ ≈ 266 mAh/g capacity, output format: ['Value', 'unit (e.g., wt.%']"),
        ResponseSchema(name="Description of the discharge curve", description="Description of the hydrogenation process (replace the content in {}), output format: The x- and y-axis units are {[x-axis unit, y-axis unit]}. The curve starts from the point {[x0, y0]}, ends at {[x1, y1]}")   
    ]
    image_parser = StructuredOutputParser.from_response_schemas(image_response_schema)
    return PromptTemplate(
        template="""        
        Figure Context:
        {caption_context}
        
        Please extract information from the following figure step by step using the provided context. Follow the exact instructions below:
        
        Step 1: Figure Identification and Contextual Summary
        Identify which figure the image corresponds to from the Figure Context (e.g., "This is Figure 3 of the article").
        Summarize the relevant figure context in plain text, including the material formula, testing condition, and any keywords related to the discharge ability, etc.
        Format: a natural paragraph. Do not output JSON here.
        
        Step 2: Curve Extraction by Subplot and Curve
        If the figure contains multiple subplots, treat each subplot independently and include a label (e.g., "subplot (a)", "subplot (b)").
        In each subplot, identify and extract every individual curve.
        If possible, use legends, colors, or arrow directions to distinguish between absorption and desorption processes.
        For every curve (not subplot), create a dictionary following the specified JSON schema: {parser_schema}.
        The final output should be a JSON-formatted list (array) of these dictionaries, with each dictionary representing a single curve.
        Please note 266 mAh/g ≈ 1 wt% H₂ capacity.
        """,
        input_variables=["caption_context"],
        partial_variables={"parser_schema": image_parser.get_format_instructions()}
    )

def get_image_tpd_prompt() -> PromptTemplate:
    image_response_schema = [
        ResponseSchema(name="ID", description="Sequential number (e.g., '1', '2'), this integer corresponds to the order of the extracted Python dictionary."),
        ResponseSchema(name="Formula", description="Chemical formula of the material or of the material prepared under different conditions. It must be a combination of element symbols and numbers."),
        ResponseSchema(
            name="Gravimetric hydrogen density", 
            description="Maximum hydrogen storage capacity Formula, along with the testing pressure and temperature. Output format: ['Value', 'unit', 'pressure value', 'pressure unit', 'temperature value', 'temperature unit']."
        ),
       ResponseSchema(
            name="Description of the hydrogenation curve",
            description=(
                "Description of the hydrogenation process (replace the content in {}). "
                "Output format: The x- and y-axis units are {[x-axis unit, y-axis unit]}. "
                "The curve starts at {[x0, y0]}, shows a rapid increase in hydrogen uptake "
                "within the range {[x1, y1]} to {[x2, y2]}, stabilizes after {[x3, y3]}, "
                "and ends at {[x4, y4]}."
            )
        ),
        ResponseSchema(
            name="Description of the dehydrogenation curve",
            description=(
                "Description of the dehydrogenation process (replace the content in {}). "
                "Output format: The x- and y-axis units are {[x-axis unit, y-axis unit]}. "
                "The curve starts at {[x0, y0]}, shows a rapid increase in hydrogen release "
                "within the range {[x1, y1]} to {[x2, y2]}, stabilizes after {[x3, y3]}, "
                "and ends at {[x4, y4]}."
            )
        )
    ]
    image_parser = StructuredOutputParser.from_response_schemas(image_response_schema)
    return PromptTemplate(
        template="""
        Figure Context:
        {caption_context}
        Please extract information from the figures step by step using the provided context. Follow these instructions exactly:
        
        Step 1: Figure Identification and Context Summary
        Identify which figure the image corresponds to in the Figure Context.
        Identify all figures that contain hydrogen absorption or desorption processes.
        Summarize the relevant figure context in plain text, including details such as material formula, temperature, pressure, and testing conditions.
        Output as a natural paragraph, not JSON.
        
        Step 2: Curve Extraction
        If the figure has multiple subplots, treat each subplot separately.
        For each figures or subplots that contain hydrogen absorption or desorption processes, identify and extract all absorption or desorption curves.
        Use legends, colors, or arrows to distinguish absorption from desorption wherever possible.
        For each curve (not subplot), generate a dictionary following the provided JSON schema: {parser_schema}.
        Output a single JSON array (list) containing all extracted curve dictionaries.
        
        Special Note:
        1. If the figure shows results from multiple cycles, only extract data for the first cycle’s absorption and desorption performance.
        """,
        input_variables=["caption_context"],
        partial_variables={"parser_schema": image_parser.get_format_instructions()}
    )

def get_text_image_prompt(with_physical = True) -> PromptTemplate:
    text_response_schema = [
        ResponseSchema(name="ID", description="Sequential number (e.g., '1', '2'), equal to the number of materials or measurement conditions reported."),
        ResponseSchema(name="doi", description="DOI (Digital Object Identifier) without the URL prefix."),    
        ResponseSchema(name="Formula", description="Chemical formula of the material or the same formula prepared under different conditions"),
        ResponseSchema(
            name="Hydrogenation temperature",
            description=(
                "For hydrogen uptake vs. time curves, record the testing temperature. "
                "For hydrogen uptake vs. temperature curves, record the end temperature of the range where most hydrogen is absorbed. "
                "Output format (list length is 2): ['Value', 'unit (K/°C)']"
            )
        ),
        ResponseSchema(name="Hydrogenation pressure", description="Equilibrium pressure during hydrogen absorption, where uptake increases rapidly and pressure rises slowly, output format (List length is 2): ['Value', 'unit (bar/MPa)']"),
        ResponseSchema(
            name="Dehydrogenation temperature",
            description=(
                "For hydrogen release (dehydrogenation) vs. time curves, record the testing temperature. "
                "For hydrogen release vs. temperature curves, record the end temperature of the range where most hydrogen is released. "
                "Output format (list length is 2): ['Value', 'unit (K/°C)']"
            )
        ),
        ResponseSchema(name="Dehydrogenation pressure", description="Equilibrium pressure during hydrogen desorption, where hydrogen release increases rapidly while pressure drops slowly, output format (List length is 2): ['Value', 'unit (bar/MPa)']"),
        ResponseSchema(name="Volumetric hydrogen capacity", description="Maximum hydrogen storage density (g H₂/L, kg H₂/m³) and its corresponding pressure and temperature, output format (List length is 6): ['Value', 'unit (g H₂/L, kg H₂/m³)', 'pressure value', 'pressure unit', 'temperature value', 'temperature unit']"),
        ResponseSchema(name="Gravimetric hydrogen density", description="Maximum hydrogen storage density (wt.%) after at least one cycle of hydrogen absorption and desorption and its corresponding pressure and temperature, output format (List length is 6): ['Value', 'unit', 'pressure value', 'pressure unit', 'temperature value', 'temperature unit']"),
        ResponseSchema(name="H2 experiment release", description="Hydrogen release amount from the start of desorption equilibrium pressure to the lowest desorption pressure, output format (List length is 2): ['value', 'unit']"), 
        ResponseSchema(name="H2 experiment adsorption", description="Hydrogen absorption amount from the lowest absorption pressure to the end of the absorption equilibrium pressure, output format (List length is 2): ['value', 'unit']"), 
        ResponseSchema(name="enthalpy change (ΔH, kJ/mol H2)", description="The enthalpy change during the hydrogen absorption and deadsorption process, output format (List length is 4): ['Value (Hydrogenation)', 'unit', 'Value (Dehydrogenation)', 'unit']"), 
        ResponseSchema(name="entropy change (ΔS, J/mol H2·K)", description="The entropy change during the hydrogen absorption and deadsorption process, output format (List length is 4): ['Value (Hydrogenation)', 'unit', 'Value (Dehydrogenation)', 'unit']"), 
        ResponseSchema(
            name="Material type",
            description=(
                "Specify the type of hydrogen storage material. Choose from the following options:\n"
                "- Interstitial Hydride (Hydrogen atoms occupy interstitial sites in metal lattices，e.g., TiFe, LaNi₅, ZrV₂, TiMn₂, ZrNi, FeTi, Mg₂Ni, VTiCr, ScAlMn, TiCrMn)\n"
                "- Complex Hydride (Contain covalently bonded anionic complexes such as [AlH₄]⁻, [BH₄]⁻, etc. e.g., NaAlH₄, LiAlH₄, LiBH₄, Mg(BH₄)₂, Ca(BH₄)₂, LiNH₂, Mg(NH₂)₂, NH₃BH₃ (ammonia borane), KAlH₄, NaZn(BH₄)₃, Li₄BN₃H₁₀, Na₂LiAlH₆)\n"
                "- Ionic Hydride (Formed by electropositive metals and H⁻ ions, typically saline-like in structure. e.g., LiH, NaH, KH, RbH, CsH, CaH₂, SrH₂, BaH₂, MgH₂, BeH₂)\n"
                "- Porous material (Hydrogen is stored by physisorption or chemisorption within high surface area frameworks. e.g., COFs (e.g., COF-1, COF-102), MOFs (e.g., MOF-5, UiO-66, HKUST-1, ZIF-8), porous carbon, activated carbon, graphene, carbon nanotubes (CNTs), zeolites)\n"
                "- Multi-component Hydride (Combinations or composites of different hydride types to improve storage properties. e.g., mixtures of interstitial, ionic or complex hydrides)\n"
                "- Superhydride (Hydrides with exceptionally high hydrogen content (typically x > 5 in MHₓ), often formed under high  (GPa). e.g. LaH10, YH9, CaH6, ThH10, ScH9, Y3Fe4H20, UH8, CeH9, AcH10, BaH12) Materials with large hydrogen content or containing hydrogen-rich complex anions (e.g., FeH₈ units) may also be referred to as superhydrides, even if they are formally categorized as complex hydrides—such as Y₃Fe₄H₂₀.\n"
                "- Others (Materials that do not clearly fit into the above categories)"
            )
        ),
        ResponseSchema(
            name="Interstitial Hydride Category",
            description=(
                "If the material type is 'Interstitial Hydride', specify its subcategory. Select one of the following (or a similar category):\n"
                "- AB2: Composed of A-site and B-site metal elements with an overall atomic ratio of approximately 1:2 (A:B). "
                "Either A or B can include multiple elements, as long as the total atomic ratio meets the AB2 requirement. "
                "Typical structures include Laves phases (e.g., TiMn2, (Ti,Zr)(Mn,Cr)2).\n"
                "- AB3: Contains A-site and B-site elements with an overall atomic ratio of about 1:3 (A:B). "
                "A or B can be multi-elemental, provided the total ratio is AB3. "
                "The structure combines AB2 (Laves polyhedron) and AB5 (capped hexagonal prism) units (e.g., LaNi3).\n"
                "- AB5: Made up of A-site and B-site metal elements with a total atomic ratio of about 1:5 (A:B). "
                "Multiple elements are allowed in A or B sites as long as the AB5 stoichiometry is satisfied. "
                "Mainly consists of capped hexagonal prism units (e.g., LaNi5, (La,Ce)(Ni,Co,Al)5).\n"
                "- Others"
            )
        ),
        ResponseSchema(
        name="physical interpretation",
        description=(
            "Provide a concise analysis of the physical mechanisms underlying changes in hydrogen storage density and equilibrium pressure. "
            "Focus on how elemental composition, content, structural features, and processing conditions influence these properties. "
            "Emphasize key mechanisms and causal relationships. "
            "Output format (replace {} with the actual material): The {Formula} exhibits this performance because..."
        )
        ),
        ResponseSchema(name="Publication year", description="Publication year, output format: YYYY (int)"), 
    ]
    if not with_physical:
        text_response_schema = [schema for schema in text_response_schema if schema.name != "physical interpretation"]

    text_parser = StructuredOutputParser.from_response_schemas(text_response_schema)
    return PromptTemplate(
        template="""
Below is the text from a scientific publication on hydrogen storage materials, along with information extracted from figures (in JSON format).  
To obtain accurate results, the text and the extracted image data can corroborate and complement each other.
The typical points extracted from the image data can also be used to verify the final performance data needed.
if the data aren't provided, output \"NA\".

Paper Text:
{paper_text}

Information extracted from diagrams:
{image_data}

For each reported material or for each distinct testing condition of one material, respond in JSON format following this schema: {parser_schema}
The number of dictionaries in the final JSON list MUST NOT be fewer than the number of dictionaries in the JSON list extracted from the images.
/think 
    """,
        input_variables=["paper_text", "image_data"],
        partial_variables={"parser_schema": text_parser.get_format_instructions()}
    )

print(get_text_image_prompt(with_physical = False).format(paper_text="Example text", image_data='[{"ID": "1", "Formula": "H2O"}]'))


def get_text_only_prompt(with_physical = True) -> PromptTemplate:
    text_response_schema = [
        ResponseSchema(name="ID", description="Sequential number (e.g., '1', '2'), equal to the number of materials or measurement conditions reported."),
        ResponseSchema(name="doi", description="DOI (Digital Object Identifier) without the URL prefix."),    
        ResponseSchema(name="Formula", description="Chemical formula of the material or the same formula prepared under different conditions"),
        ResponseSchema(
            name="Hydrogenation temperature",
            description=(
                "For hydrogen uptake vs. time curves, record the testing temperature. "
                "For hydrogen uptake vs. temperature curves, record the end temperature of the range where most hydrogen is absorbed. "
                "Output format (list length is 2): ['Value', 'unit (K/°C)']"
            )
        ),
        ResponseSchema(name="Hydrogenation pressure", description="Equilibrium pressure during hydrogen absorption, where uptake increases rapidly and pressure rises slowly, output format (List length is 2): ['Value', 'unit (bar/MPa)']"),
        ResponseSchema(
            name="Dehydrogenation temperature",
            description=(
                "For hydrogen release (dehydrogenation) vs. time curves, record the testing temperature. "
                "For hydrogen release vs. temperature curves, record the end temperature of the range where most hydrogen is released. "
                "Output format (list length is 2): ['Value', 'unit (K/°C)']"
            )
        ),
        ResponseSchema(name="Dehydrogenation pressure", description="Equilibrium pressure during hydrogen desorption, where hydrogen release increases rapidly while pressure drops slowly, output format (List length is 2): ['Value', 'unit (bar/MPa)']"),
        ResponseSchema(name="Volumetric hydrogen capacity", description="Maximum hydrogen storage density (g H₂/L, kg H₂/m³) and its corresponding pressure and temperature, output format (List length is 6): ['Value', 'unit (g H₂/L, kg H₂/m³)', 'pressure value', 'pressure unit', 'temperature value', 'temperature unit']"),
        ResponseSchema(name="Gravimetric hydrogen density", description="Maximum hydrogen storage density (wt.%) after at least one cycle of hydrogen absorption and desorption and its corresponding pressure and temperature, output format (List length is 6): ['Value', 'unit', 'pressure value', 'pressure unit', 'temperature value', 'temperature unit']"),
        ResponseSchema(name="H2 experiment release", description="Hydrogen release amount from the start of desorption equilibrium pressure to the lowest desorption pressure, output format (List length is 2): ['value', 'unit']"), 
        ResponseSchema(name="H2 experiment adsorption", description="Hydrogen absorption amount from the lowest absorption pressure to the end of the absorption equilibrium pressure, output format (List length is 2): ['value', 'unit']"), 
        ResponseSchema(name="enthalpy change (ΔH, kJ/mol H2)", description="The enthalpy change during the hydrogen absorption and deadsorption process, output format (List length is 4): ['Value (Hydrogenation)', 'unit', 'Value (Dehydrogenation)', 'unit']"), 
        ResponseSchema(name="entropy change (ΔS, J/mol H2·K)", description="The entropy change during the hydrogen absorption and deadsorption process, output format (List length is 4): ['Value (Hydrogenation)', 'unit', 'Value (Dehydrogenation)', 'unit']"), 
        ResponseSchema(
            name="Material type",
            description=(
                "Specify the type of hydrogen storage material. Choose from the following options:\n"
                "- Interstitial Hydride (Hydrogen atoms occupy interstitial sites in metal lattices，e.g., TiFe, LaNi₅, ZrV₂, TiMn₂, ZrNi, FeTi, Mg₂Ni, VTiCr, ScAlMn, TiCrMn)\n"
                "- Complex Hydride (Contain covalently bonded anionic complexes such as [AlH₄]⁻, [BH₄]⁻, etc. e.g., NaAlH₄, LiAlH₄, LiBH₄, Mg(BH₄)₂, Ca(BH₄)₂, LiNH₂, Mg(NH₂)₂, NH₃BH₃ (ammonia borane), KAlH₄, NaZn(BH₄)₃, Li₄BN₃H₁₀, Na₂LiAlH₆)\n"
                "- Ionic Hydride (Formed by electropositive metals and H⁻ ions, typically saline-like in structure. e.g., LiH, NaH, KH, RbH, CsH, CaH₂, SrH₂, BaH₂, MgH₂, BeH₂)\n"
                "- Porous material (Hydrogen is stored by physisorption or chemisorption within high surface area frameworks. e.g., COFs (e.g., COF-1, COF-102), MOFs (e.g., MOF-5, UiO-66, HKUST-1, ZIF-8), porous carbon, activated carbon, graphene, carbon nanotubes (CNTs), zeolites)\n"
                "- Multi-component Hydride (Combinations or composites of different hydride types to improve storage properties. e.g., mixtures of interstitial, ionic or complex hydrides)\n"
                "- Superhydride (Hydrides with exceptionally high hydrogen content (typically x > 5 in MHₓ), often formed under high  (GPa). e.g. LaH10, YH9, CaH6, ThH10, ScH9, Y3Fe4H20, UH8, CeH9, AcH10, BaH12) Materials with large hydrogen content or containing hydrogen-rich complex anions (e.g., FeH₈ units) may also be referred to as superhydrides, even if they are formally categorized as complex hydrides—such as Y₃Fe₄H₂₀.\n"
                "- Others (Materials that do not clearly fit into the above categories)"
            )
        ),
        ResponseSchema(
            name="Interstitial Hydride Category",
            description=(
                "If the material type is 'Interstitial Hydride', specify its subcategory. Select one of the following (or a similar category):\n"
                "- AB2: Composed of A-site and B-site metal elements with an overall atomic ratio of approximately 1:2 (A:B). "
                "Either A or B can include multiple elements, as long as the total atomic ratio meets the AB2 requirement. "
                "Typical structures include Laves phases (e.g., TiMn2, (Ti,Zr)(Mn,Cr)2).\n"
                "- AB3: Contains A-site and B-site elements with an overall atomic ratio of about 1:3 (A:B). "
                "A or B can be multi-elemental, provided the total ratio is AB3. "
                "The structure combines AB2 (Laves polyhedron) and AB5 (capped hexagonal prism) units (e.g., LaNi3).\n"
                "- AB5: Made up of A-site and B-site metal elements with a total atomic ratio of about 1:5 (A:B). "
                "Multiple elements are allowed in A or B sites as long as the AB5 stoichiometry is satisfied. "
                "Mainly consists of capped hexagonal prism units (e.g., LaNi5, (La,Ce)(Ni,Co,Al)5).\n"
                "- Others"
            )
        ),
        ResponseSchema(
        name="physical interpretation",
        description=(
            "Provide a concise analysis of the physical mechanisms underlying changes in hydrogen storage density and equilibrium pressure. "
            "Focus on how elemental composition, content, structural features, and processing conditions influence these properties. "
            "Emphasize key mechanisms and causal relationships. "
            "Output format (replace {} with the actual material): The {Formula} exhibits this performance because..."
        )
        ),
        ResponseSchema(name="Publication year", description="Publication year, output format: YYYY (int)"), 
    ]
    if not with_physical:
        text_response_schema = [schema for schema in text_response_schema if schema.name != "physical interpretation"]
    text_parser = StructuredOutputParser.from_response_schemas(text_response_schema)
    return PromptTemplate(
        template="""
Below is the text from a scientific publication on hydrogen storage materials, along with information extracted from figures (in JSON format).  
To obtain accurate results, the text and the extracted image data can corroborate and complement each other.
The typical points extracted from the image data can also be used to verify the final performance data needed.
if the data aren't provided, output \"NA\".

Paper Text:
{paper_text}

For each reported material or for each distinct testing condition of one material, respond in JSON format following this schema: {parser_schema}
The number of dictionaries in the final JSON list MUST NOT be fewer than the number of dictionaries in the JSON list extracted from the images.
/think
""",
        input_variables=["paper_text"],
        partial_variables={"parser_schema": text_parser.get_format_instructions()}
    )

def judge_formula_prompt() -> PromptTemplate:
    return PromptTemplate(
        template="""
You will be given a string representing a chemical formula (formula) for a material.
Please determine whether this formula correctly reflects both the types of elements and their compositional ratios in the material.

Criteria:
- The formula must contain at least two different chemical elements, and the ratio of each element should be clear (e.g., MgH2, LaNi5, NaAlH4).
- If the formula contains only one type of element (e.g., H2, O2, Fe), answer "no".
- Only answer "yes" or "no". Do not provide any explanation.

Formula: {text}
Your answer (only "yes" or "no"):
""",
        input_variables=["text"],
    )



def matching_formula_prompt() -> PromptTemplate:
    return PromptTemplate(
        template="""
You are given a list of chemical formulas (formula_list) and the full text content of a scientific paper (markdown_content).

Instructions:
- For each formula in formula_list, carefully analyze the Scientific paper content below to identify all constituent elements present in the reported formula.
- List each element using its standard chemical symbol (e.g., Mg, Li, Ca, La, Ni, Mn), separated by commas.
- If any element in the formula cannot be identified from the markdown_content, return the original formula as the value.
- Output a JSON dictionary where each key is the original formula from formula_list, and the value is the complete formula or the identified elements and their ratios.

Example output:
{{
    "Mg-Li-Ca": "Mg, Li, Ca",
    "LaNi5": "La, Ni",
    "MmNi5": "La, Mg, Al, Ni"
}}

formula_list: {formula_list}

Scientific paper content:
{markdown_content}
/think
""",
        input_variables=["formula_list", "markdown_content"],
    )


def standard_formula_prompt() -> PromptTemplate:
    return PromptTemplate(
        template="""
You are given a list of formulas and the full text content of a scientific paper (markdown_content).

Instructions:
- Analyze the atomic ratio of each element in the provided formula list and reconstruct the standard chemical formula.
- If the element ratio is not clear, return the "NA" value.
- Output a JSON dictionary where each key is the formula, and the value is the complete formula including the elements and their atomic ratios.
- The atomic ratios should be represented as subscripts and the subscripts should be placed after the element symbols (e.g., Mg0.2Li0.3Ca0.5).
- The subscript sums should equal 1.0, and the order of elements in the formula should match the order in the element list.

Example output:
{{
    "Mg-Li-Ca": "Mg0.2Li0.3Ca0.5",
    "LaNi5": "La0.2Ni0.8",
    "MmNi5": "La0.1Mg0.2Al0.3Ni0.4"
}}

formula_list: {formula_list}

Scientific paper content:
{markdown_content}
/think
""",
        input_variables=["element_list", "markdown_content"],
    )


def get_prompt_template(template_name: str) -> Any:
    """
    获取指定名称的 PromptTemplate。
    支持: 'material_type', 'reformat', 'image_pct', 'image_discharge', 'image_tpd', 'text_image', 'text_only'，后续可扩展。
    """
    if template_name == 'material_type':
        return get_material_type_template()
    elif template_name == 'reformat':
        return get_reformat_prompt()
    elif template_name == 'image_pct':
        return get_image_pct_prompt()
    elif template_name == 'image_discharge':
        return get_image_discharge_prompt()
    elif template_name == 'image_tpd':
        return get_image_tpd_prompt()
    elif template_name == 'text_image':
        return get_text_image_prompt()
    elif template_name == 'text_only':
        return get_text_only_prompt()
    else:
        raise ValueError(f"Unknown template name: {template_name}")