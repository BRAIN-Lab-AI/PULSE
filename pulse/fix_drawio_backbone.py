"""
Modifies the draw.io XML to replace the 12 individual MHSA+FFN blocks
in the backbone container with 4 grouped blocks.

Usage:
  1. In draw.io: Extras -> Edit Diagram -> copy all XML -> paste into a .xml file
     (or File -> Export As -> XML)
  2. Run: python fix_drawio_backbone.py your_file.xml
  3. Open the output file (your_file_modified.xml) in draw.io
"""

import sys
import xml.etree.ElementTree as ET

BACKBONE_ID = "7spE5FbKINC6AD-i8wyv-156"

NEW_GROUPS = [
    {"id": "mhsa_grp1", "y": 380, "fill": "#fde68a"},  # bottom — lightest
    {"id": "mhsa_grp2", "y": 255, "fill": "#fde68a"},
    {"id": "mhsa_grp3", "y": 130, "fill": "#fbbf24"},
    {"id": "mhsa_grp4", "y":   5, "fill": "#f59e0b"},  # top — darkest
]

def fix(input_path):
    ET.register_namespace("", "")
    tree = ET.parse(input_path)
    root = tree.getroot()

    # draw.io XML structure: <mxGraphModel><root><mxCell .../>...</root></mxGraphModel>
    xml_root = root.find("root")
    if xml_root is None:
        print("ERROR: could not find <root> element — is this a draw.io XML file?")
        sys.exit(1)

    # --- Remove old MHSA+FFN cells inside the backbone container ---
    removed = 0
    for cell in list(xml_root):
        parent = cell.get("parent", "")
        value  = cell.get("value", "")
        if parent == BACKBONE_ID and "MULTI HEAD" in value.upper():
            xml_root.remove(cell)
            removed += 1

    print(f"Removed {removed} old MHSA+FFN rows from backbone container.")

    # --- Add 4 new grouped cells ---
    for g in NEW_GROUPS:
        cell = ET.SubElement(xml_root, "mxCell")
        cell.set("id",     g["id"])
        cell.set("value",  "MHSA + FFN")
        cell.set("style",
                 f"rounded=1;"
                 f"fillColor={g['fill']};"
                 f"strokeColor=#d97706;"
                 f"fontSize=11;"
                 f"fontStyle=1;"
                 f"align=center;")
        cell.set("vertex", "1")
        cell.set("parent", BACKBONE_ID)

        geo = ET.SubElement(cell, "mxGeometry")
        geo.set("x",      "10")
        geo.set("y",      str(g["y"]))
        geo.set("width",  "260")
        geo.set("height", "55")
        geo.set("as",     "geometry")

    print("Added 4 new grouped MHSA+FFN blocks (bottom→top: lightest→darkest amber).")

    # --- Save ---
    output_path = input_path.replace(".xml", "_modified.xml")
    tree.write(output_path, encoding="unicode", xml_declaration=False)
    print(f"\nSaved: {output_path}")
    print("Open that file in draw.io (Extras -> Edit Diagram -> paste content, or File -> Import).")
    print("\nNOTE: if the 4 blocks don't fit neatly inside the backbone box,")
    print("adjust the y-values (5, 130, 255, 380) in this script to match your box height.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_drawio_backbone.py your_diagram.xml")
        sys.exit(1)
    fix(sys.argv[1])
