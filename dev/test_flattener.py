import json
import urllib.request
from pathlib import Path

def flatten_workflow(wf_data: dict) -> dict:
    subgraph_map = {sg["id"]: sg for sg in wf_data.get("definitions", {}).get("subgraphs", [])}
    
    api_prompt = {}

    # Map top links: link_id -> dict
    top_links = {}
    for link in wf_data.get("links", []):
        top_links[link[0]] = {
            "id": link[0],
            "origin_id": link[1],
            "origin_slot": link[2],
            "target_id": link[3],
            "target_slot": link[4],
            "type": link[5]
        }

    # Helper to resolve origin for top-level link
    def resolve_top_origin(origin_id, origin_slot):
        origin_node = next((n for n in wf_data.get("nodes", []) if n["id"] == origin_id), None)
        if not origin_node:
            return None
        
        # If it's a Reroute node
        if origin_node["type"] == "Reroute":
            # find what feeds this reroute
            for inp in origin_node.get("inputs", []):
                lid = inp.get("link")
                if lid and lid in top_links:
                    return resolve_top_origin(top_links[lid]["origin_id"], top_links[lid]["origin_slot"])
            return None

        # Check if origin_node is a subgraph instance
        if origin_node["type"] in subgraph_map:
            sg = subgraph_map[origin_node["type"]]
            sg_links = {l["id"]: l for l in sg.get("links", [])}
            # Find which internal node connects to Out[origin_slot]
            out_desc = sg["outputs"][origin_slot]
            for lid in out_desc.get("linkIds", []):
                if lid in sg_links:
                    ilink = sg_links[lid]
                    if ilink["target_id"] == -20 and ilink["target_slot"] == origin_slot:
                        # Trace back through any internal Reroutes
                        return resolve_internal_origin(origin_node, sg, ilink["origin_id"], ilink["origin_slot"])
            return None
        else:
            return [str(origin_id), origin_slot]

    # Helper to resolve internal origin inside a subgraph
    def resolve_internal_origin(sg_instance_node, sg, origin_id, origin_slot):
        sg_links = {l["id"]: l for l in sg.get("links", [])}
        
        if origin_id == -10:
            # Sourced from subgraph input slot origin_slot
            # Check what connects to this subgraph input on the top level
            instance_inputs = sg_instance_node.get("inputs", [])
            instance_wv = sg_instance_node.get("widgets_values") or []
            if origin_slot < len(instance_inputs):
                s_inp = instance_inputs[origin_slot]
                lid = s_inp.get("link")
                if lid and lid in top_links:
                    t_link = top_links[lid]
                    return resolve_top_origin(t_link["origin_id"], t_link["origin_slot"])
                else:
                    # Widget constant value!
                    unlinked_before = sum(1 for prev in instance_inputs[:origin_slot] if prev.get("link") is None)
                    if unlinked_before < len(instance_wv):
                        return instance_wv[unlinked_before]
            return None

        inode = next((n for n in sg.get("nodes", []) if n["id"] == origin_id), None)
        if not inode:
            return None

        if inode["type"] == "Reroute":
            # Follow Reroute input
            for inp in inode.get("inputs", []):
                lid = inp.get("link")
                if lid and lid in sg_links:
                    ilink = sg_links[lid]
                    return resolve_internal_origin(sg_instance_node, sg, ilink["origin_id"], ilink["origin_slot"])
            return None

        return [f"{sg_instance_node['id']}_{origin_id}", origin_slot]

    # Process top-level nodes
    for node in wf_data.get("nodes", []):
        nid = node["id"]
        ntype = node["type"]
        
        if ntype in ["MarkdownNote", "Note", "PreviewAny", "Reroute"]:
            continue

        if ntype in subgraph_map:
            sg = subgraph_map[ntype]
            sg_links = {l["id"]: l for l in sg.get("links", [])}
            instance_inputs = node.get("inputs", [])
            instance_wv = node.get("widgets_values") or []

            # Create each internal node of the subgraph
            for inode in sg.get("nodes", []):
                in_type = inode["type"]
                if in_type in ["MarkdownNote", "Note", "PreviewAny", "Reroute"]:
                    continue
                in_id = f"{nid}_{inode['id']}"
                in_inputs = {}
                
                inode_inputs = inode.get("inputs", [])
                inode_wv = inode.get("widgets_values") or []

                for in_s_idx, in_inp in enumerate(inode_inputs):
                    inp_name = in_inp["name"]
                    in_lid = in_inp.get("link")
                    if in_lid and in_lid in sg_links:
                        ilink = sg_links[in_lid]
                        resolved = resolve_internal_origin(node, sg, ilink["origin_id"], ilink["origin_slot"])
                        if resolved is not None:
                            in_inputs[inp_name] = resolved
                    else:
                        unlinked_before = sum(1 for prev in inode_inputs[:in_s_idx] if prev.get("link") is None)
                        if unlinked_before < len(inode_wv):
                            in_inputs[inp_name] = inode_wv[unlinked_before]

                api_prompt[in_id] = {
                    "class_type": in_type,
                    "inputs": in_inputs,
                    "_meta": {"title": inode.get("title") or in_type}
                }
        else:
            # Regular top-level node
            top_in_inputs = {}
            node_inputs = node.get("inputs", [])
            node_wv = node.get("widgets_values") or []

            if ntype == "PrimitiveStringMultiline":
                if node_wv:
                    top_in_inputs["value"] = node_wv[0]
            elif ntype == "LoadVideo":
                if node_wv:
                    top_in_inputs["file"] = node_wv[0]
            elif ntype == "SaveVideo":
                # Find inputs
                for s_idx, s_inp in enumerate(node_inputs):
                    inp_name = s_inp["name"]
                    lid = s_inp.get("link")
                    if lid and lid in top_links:
                        t_link = top_links[lid]
                        res = resolve_top_origin(t_link["origin_id"], t_link["origin_slot"])
                        if res is not None:
                            top_in_inputs[inp_name] = res
                if len(node_wv) >= 1:
                    top_in_inputs["filename_prefix"] = node_wv[0]
                if len(node_wv) >= 2:
                    top_in_inputs["format"] = node_wv[1]
                if len(node_wv) >= 3 and node_wv[2] != "auto":
                    top_in_inputs["codec"] = node_wv[2]
            else:
                for s_idx, s_inp in enumerate(node_inputs):
                    inp_name = s_inp["name"]
                    lid = s_inp.get("link")
                    if lid and lid in top_links:
                        t_link = top_links[lid]
                        res = resolve_top_origin(t_link["origin_id"], t_link["origin_slot"])
                        if res is not None:
                            top_in_inputs[inp_name] = res
                    else:
                        unlinked_before = sum(1 for prev in node_inputs[:s_idx] if prev.get("link") is None)
                        if unlinked_before < len(node_wv):
                            top_in_inputs[inp_name] = node_wv[unlinked_before]

            api_prompt[str(nid)] = {
                "class_type": ntype,
                "inputs": top_in_inputs,
                "_meta": {"title": node.get("title") or ntype}
            }

    return api_prompt

if __name__ == "__main__":
    with open("dev/workflows/LTX-2.5_V2V_TiledFusion_Restore.json", "r", encoding="utf-8") as f:
        wf = json.load(f)

    prompt = flatten_workflow(wf)
    print(f"Generated API prompt with {len(prompt)} nodes.")
    with open("test_flattened_prompt.json", "w", encoding="utf-8") as f:
        json.dump(prompt, f, indent=2)

    # Validate against ComfyUI /prompt endpoint
    req = urllib.request.Request(
        "http://127.0.0.1:8188/prompt",
        data=json.dumps({"prompt": prompt, "client_id": "test_validator"}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            print("Response:", resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print("HTTPError:", e.code, e.read().decode("utf-8"))
