from typing import Any, TypedDict
import re


class AccessibilityTreeNode(TypedDict):
    nodeId: str
    ignored: bool
    role: dict[str, Any]
    chromeRole: dict[str, Any]
    name: dict[str, Any]
    properties: list[dict[str, Any]]
    childIds: list[str]
    parentId: str
    backendDOMNodeId: str
    frameId: str
    bound: list[float] | None
    union_bound: list[float] | None
    offsetrect_bound: list[float] | None


class BrowserConfig(TypedDict):
    win_top_bound: float
    win_left_bound: float
    win_width: float
    win_height: float
    win_right_bound: float
    win_lower_bound: float
    device_pixel_ratio: float


class BrowserInfo(TypedDict):
    DOMTree: dict[str, Any]
    config: BrowserConfig

IGNORED_ACTREE_PROPERTIES = (
    "focusable",
    "editable",
    "readonly",
    "level",
    "settable",
    "multiline",
    "invalid",
    "hiddenRoot",
    "hidden",
    "controls",
    "labelledby",
    "describedby",
)

AccessibilityTree = list[AccessibilityTreeNode]

IN_VIEWPORT_RATIO_THRESHOLD = 0.8



def fetch_browser_info(
    # page: Page,
    browser,
) -> BrowserInfo:
    # extract domtree
    tree = browser.execute_cdp_cmd(
        "DOMSnapshot.captureSnapshot",
        {
            "computedStyles": [],
            "includeDOMRects": True,
            "includePaintOrder": True,
        },
    )

    # calibrate the bounds, in some cases, the bounds are scaled somehow
    bounds = tree["documents"][0]["layout"]["bounds"]
    b = bounds[0]
    n = b[2] / browser.get_window_size()["width"]
    bounds = [[x / n for x in bound] for bound in bounds]
    tree["documents"][0]["layout"]["bounds"] = bounds

    # extract browser info
    # win_top_bound = page.evaluate("window.pageYOffset")
    # win_left_bound = page.evaluate("window.pageXOffset")
    # win_width = page.evaluate("window.screen.width")
    # win_height = page.evaluate("window.screen.height")

    win_top_bound = browser.execute_script("return window.pageYOffset;")
    win_left_bound = browser.execute_script("return window.pageXOffset;")
    # win_width = browser.execute_script("return window.screen.width;")
    # win_height = browser.execute_script("return window.screen.height;")
    win_width = browser.execute_script("return window.innerWidth;")
    win_height = browser.execute_script("return window.innerHeight;")
    win_right_bound = win_left_bound + win_width
    win_lower_bound = win_top_bound + win_height
    device_pixel_ratio = browser.execute_script("return window.devicePixelRatio;")
    assert device_pixel_ratio == 1.0, "devicePixelRatio is not 1.0"

    config: BrowserConfig = {
        "win_top_bound": win_top_bound,
        "win_left_bound": win_left_bound,
        "win_width": win_width,
        "win_height": win_height,
        "win_right_bound": win_right_bound,
        "win_lower_bound": win_lower_bound,
        "device_pixel_ratio": device_pixel_ratio,
    }
    # print (config)
    # assert len(tree['documents']) == 1, "More than one document in the DOM tree"
    info: BrowserInfo = {"DOMTree": tree, "config": config}

    return info




def get_element_in_viewport_ratio(
    elem_left_bound: float,
    elem_top_bound: float,
    width: float,
    height: float,
    config: BrowserConfig,
) -> float:
    elem_right_bound = elem_left_bound + width
    elem_lower_bound = elem_top_bound + height

    win_left_bound = 0
    win_right_bound = config["win_width"]
    win_top_bound = 0
    win_lower_bound = config["win_height"]

    # Compute the overlap in x and y axes
    overlap_width = max(
        0,
        min(elem_right_bound, win_right_bound)
        - max(elem_left_bound, win_left_bound),
    )
    overlap_height = max(
        0,
        min(elem_lower_bound, win_lower_bound)
        - max(elem_top_bound, win_top_bound),
    )

    # Compute the overlap area
    ratio = overlap_width * overlap_height / width * height
    return ratio




def get_bounding_client_rect(
    browser, backend_node_id: str
) -> dict[str, Any]:
    try:
        remote_object = browser.execute_cdp_cmd(
            "DOM.resolveNode", {"backendNodeId": int(backend_node_id)}
        )
        remote_object_id = remote_object["object"]["objectId"]
        response = browser.execute_cdp_cmd(
            "Runtime.callFunctionOn",
            {
                "objectId": remote_object_id,
                "functionDeclaration": """
                    function() {
                        if (this.nodeType == 3) {
                            var range = document.createRange();
                            range.selectNode(this);
                            var rect = range.getBoundingClientRect().toJSON();
                            range.detach();
                            return rect;
                        } else {
                            return this.getBoundingClientRect().toJSON();
                        }
                    }
                """,
                "returnByValue": True,
            },
        )
        return response
    except:
        return {"result": {"subtype": "error"}}


def fetch_page_accessibility_tree(
    info: BrowserInfo,
    browser,
    # client: CDPSession,
    current_viewport_only: bool,
) -> AccessibilityTree:
    accessibility_tree: AccessibilityTree = browser.execute_cdp_cmd(
        "Accessibility.getFullAXTree", {}
    )["nodes"]

    # a few nodes are repeated in the accessibility tree
    seen_ids = set()
    _accessibility_tree = []
    for node in accessibility_tree:
        if node["nodeId"] not in seen_ids:
            _accessibility_tree.append(node)
            seen_ids.add(node["nodeId"])
    accessibility_tree = _accessibility_tree

    nodeid_to_cursor = {}
    for cursor, node in enumerate(accessibility_tree):
        nodeid_to_cursor[node["nodeId"]] = cursor
        # usually because the node is not visible etc
        if "backendDOMNodeId" not in node:
            node["union_bound"] = None
            continue
        backend_node_id = str(node["backendDOMNodeId"])
        if node["role"]["value"] == "RootWebArea":
            # always inside the viewport
            node["union_bound"] = [0.0, 0.0, 10.0, 10.0]
        else:
            response = get_bounding_client_rect(
                browser, backend_node_id
            )
            if response.get("result", {}).get("subtype", "") == "error":
                node["union_bound"] = None
            else:
                x = response["result"]["value"]["x"]
                y = response["result"]["value"]["y"]
                width = response["result"]["value"]["width"]
                height = response["result"]["value"]["height"]
                node["union_bound"] = [x, y, width, height]

    # filter nodes that are not in the current viewport
    if current_viewport_only:

        def remove_node_in_graph(node: AccessibilityTreeNode) -> None:
            # update the node information in the accessibility tree
            nodeid = node["nodeId"]
            node_cursor = nodeid_to_cursor[nodeid]
            parent_nodeid = node["parentId"]
            children_nodeids = node["childIds"]
            parent_cursor = nodeid_to_cursor[parent_nodeid]
            # update the children of the parent node
            assert (
                accessibility_tree[parent_cursor].get("parentId", "Root")
                is not None
            )
            # remove the nodeid from parent's childIds
            index = accessibility_tree[parent_cursor]["childIds"].index(
                nodeid
            )
            accessibility_tree[parent_cursor]["childIds"].pop(index)
            # Insert children_nodeids in the same location
            for child_nodeid in children_nodeids:
                accessibility_tree[parent_cursor]["childIds"].insert(
                    index, child_nodeid
                )
                index += 1
            # update children node's parent
            for child_nodeid in children_nodeids:
                child_cursor = nodeid_to_cursor[child_nodeid]
                accessibility_tree[child_cursor][
                    "parentId"
                ] = parent_nodeid
            # mark as removed
            accessibility_tree[node_cursor]["parentId"] = "[REMOVED]"

        config = info["config"]
        for node in accessibility_tree:
            if not node["union_bound"]:
                remove_node_in_graph(node)
                continue

            [x, y, width, height] = node["union_bound"]

            # invisible node
            if width == 0 or height == 0:
                remove_node_in_graph(node)
                continue

            in_viewport_ratio = get_element_in_viewport_ratio(
                elem_left_bound=float(x),
                elem_top_bound=float(y),
                width=float(width),
                height=float(height),
                config=config,
            )

            if in_viewport_ratio < IN_VIEWPORT_RATIO_THRESHOLD:
                remove_node_in_graph(node)

        accessibility_tree = [
            node
            for node in accessibility_tree
            if node.get("parentId", "Root") != "[REMOVED]"
        ]

    return accessibility_tree


def parse_accessibility_tree(
    accessibility_tree: AccessibilityTree,
) -> tuple[str, dict[str, Any]]:
    """Parse the accessibility tree into a string text"""
    node_id_to_idx = {}
    for idx, node in enumerate(accessibility_tree):
        node_id_to_idx[node["nodeId"]] = idx

    # obs_nodes_info = {}
    nodeIdToTreeIds = {}
    def dfs(idx: int, depth: int, parent_name: str) -> str:
        tree_str = ""
        node = accessibility_tree[idx]
        indent = "\t" * depth
        valid_node = True
        try:
            role = node["role"]["value"]
            name = node["name"]["value"]
            node_str = f"{role} {repr(name)}"
            if not name.strip() or role in ['gridcell'] or (name.strip() in parent_name and role in ['StaticText', 'heading', 'image', 'generic']):
                valid_node = False
            else:
                properties = []
                for property in node.get("properties", []):
                    try:
                        if property["name"] in IGNORED_ACTREE_PROPERTIES:
                            continue
                        properties.append(
                            f'{property["name"]}: {property["value"]["value"]}'
                        )
                    except KeyError:
                        pass

                if properties:
                    node_str += " " + " ".join(properties)

                # # check valid
                # if not node_str.strip():
                #     valid_node = False

                # # empty generic node
                # if not name.strip():
                #     if not properties:
                #         if role in [
                #             "generic",
                #             "img",
                #             "list",
                #             "strong",
                #             "paragraph",
                #             "banner",
                #             "navigation",
                #             "Section",
                #             "LabelText",
                #             "Legend",
                #             "listitem",
                #         ]:
                #             valid_node = False
                #     elif role in ["listitem"]:
                #         valid_node = False

            if valid_node:
                nodeIdToTreeIds[len(nodeIdToTreeIds)+1] = node
                tree_str += f"{indent}[{len(nodeIdToTreeIds)}] {node_str}"
                # print (f"{indent}[{len(nodeIdToTreeIds)}] {node_str}", node["union_bound"])
                # obs_nodes_info[len(nodeIdToTreeIds)] = {
                #     "backend_id": node["backendDOMNodeId"],
                #     "union_bound": node["union_bound"],
                #     "text": node_str,
                # }

        except:
            valid_node = False

        for _, child_node_id in enumerate(node["childIds"]):
            if child_node_id not in node_id_to_idx:
                continue
            if len(nodeIdToTreeIds) > 300:
                break
            # mark this to save some tokens
            child_depth = depth + 1 if valid_node else depth
            curr_name = name if valid_node else parent_name
            child_str = dfs(
                node_id_to_idx[child_node_id], child_depth, curr_name
            )
            if child_str.strip():
                if tree_str.strip():
                    tree_str += "\n"
                tree_str += child_str

        return tree_str

    tree_str = dfs(0, 0, 'root')
    return tree_str, nodeIdToTreeIds


def clean_accesibility_tree(tree_str: str) -> str:
    """further clean accesibility tree"""
    clean_idx = []
    pattern = r'^\[\d+\]\s(\w+)\s\'([^\']+)\''
    prev_role = None 
    prev_name = None
    all_lines = tree_str.split('\n')
    for i, line in enumerate(all_lines):
        match = re.match(pattern, line.strip())
        if prev_role:
            if match:
                role, name = match.groups()
                if role == prev_role and (name in prev_name or prev_name in name):
                    if len(line) >= len(all_lines[clean_idx[-1]]):
                        clean_idx.pop()
                    else:
                        continue
        clean_idx.append(i)
        if match:
            prev_role, prev_name = match.groups()
    return "\n".join([all_lines[i] for i in clean_idx])


a = """[1] RootWebArea 'Google Flights - Find Cheap Flight Options & Track Prices' focused: True
	[2] button 'Main menu' expanded: False
	[3] link 'Google'
	[4] button 'Skip to main content'
	[5] button 'Accessibility feedback' hasPopup: dialog
	[6] link 'Travel'
	[7] link 'Explore'
	[8] link 'Flights'
	[9] link 'Hotels'
	[10] link 'Vacation rentals'
	[11] button 'Change appearance' hasPopup: menu expanded: False
	[12] button 'Google apps' expanded: False
	[13] link 'Sign in'
	[14] search 'Flight'
		[15] combobox 'Change ticket type. \u200bRound trip' hasPopup: listbox required: False expanded: False
		[16] button '1 passenger' hasPopup: dialog
		[17] combobox 'Change seating class. \u200bEconomy' hasPopup: listbox required: False expanded: False
		[18] combobox 'Where from?' autocomplete: inline hasPopup: menu required: False expanded: False
		[19] button 'Swap origin and destination.'
		[20] combobox 'Where to?' autocomplete: inline hasPopup: menu required: False expanded: False
		[21] textbox 'Departure' required: False
        [23] textbox 'Departure' focused: True required: False
			[24] StaticText 'Wed, Jan 1'
		[22] textbox 'Return' required: False
		[25] textbox 'Return' required: False
		[26] button 'Reset'
		[27] button 'Friday, December 6, 2024'
		[28] button 'Saturday, December 7, 2024'
		[29] button 'Friday, December 13, 2024'
		[30] button 'Saturday, December 14, 2024'
		[31] button 'Friday, December 20, 2024'
		[32] button 'Saturday, December 21, 2024'
		[33] button 'Friday, December 27, 2024'
		[34] button 'Saturday, December 28, 2024'
		[35] StaticText 'January 2025'
		[36] button 'Wednesday, January 1, 2025, departure date. , 466 US dollars'
			[37] StaticText '$466'
		[38] button 'Thursday, January 2, 2025 , 490 US dollars'
			[39] StaticText '$490'
		[40] button 'Friday, January 3, 2025 , 489 US dollars'
			[41] StaticText '$489'
		[42] button 'Saturday, January 4, 2025 , 466 US dollars'
			[43] StaticText '$466'
		[44] button 'Sunday, January 5, 2025 , 466 US dollars'
			[45] StaticText '$466'
		[46] button 'Monday, January 6, 2025 , 460 US dollars'
			[47] StaticText '$460'
		[48] button 'Tuesday, January 7, 2025 , 460 US dollars'
			[49] StaticText '$460'
		[50] button 'Wednesday, January 8, 2025 , 460 US dollars'
			[51] StaticText '$460'
		[52] button 'Thursday, January 9, 2025 , 460 US dollars'
			[53] StaticText '$460'
		[54] button 'Friday, January 10, 2025 , 460 US dollars'
			[55] StaticText '$460'
		[56] button 'Saturday, January 11, 2025 , 460 US dollars'
			[57] StaticText '$460'
		[58] button 'Sunday, January 12, 2025 , 460 US dollars'
			[59] StaticText '$460'
		[60] button 'Monday, January 13, 2025 , 460 US dollars'
			[61] StaticText '$460'
		[62] button 'Tuesday, January 14, 2025 , 460 US dollars'
			[63] StaticText '$460'
		[64] button 'Wednesday, January 15, 2025 , 460 US dollars'
			[65] StaticText '$460'
		[66] button 'Thursday, January 16, 2025 , 460 US dollars'
			[67] StaticText '$460'
		[68] button 'Friday, January 17, 2025 , 460 US dollars'
			[69] StaticText '$460'
		[70] button 'Saturday, January 18, 2025 , 460 US dollars'
			[71] StaticText '$460'
		[72] button 'Sunday, January 19, 2025 , 460 US dollars'
			[73] StaticText '$460'
		[74] button 'Monday, January 20, 2025 , 460 US dollars'
			[75] StaticText '$460'
		[76] button 'Tuesday, January 21, 2025 , 460 US dollars'
			[77] StaticText '$460'
		[78] button 'Wednesday, January 22, 2025 , 460 US dollars'
			[79] StaticText '$460'
		[80] button 'Thursday, January 23, 2025 , 460 US dollars'
			[81] StaticText '$460'
		[82] button 'Friday, January 24, 2025 , 460 US dollars'
			[83] StaticText '$460'
		[84] button 'Saturday, January 25, 2025 , 460 US dollars'
			[85] StaticText '$460'
		[86] button 'Sunday, January 26, 2025 , 460 US dollars'
			[87] StaticText '$460'
		[88] button 'Monday, January 27, 2025 , 460 US dollars'
			[89] StaticText '$460'
		[90] button 'Tuesday, January 28, 2025 , 460 US dollars'
			[91] StaticText '$460'
		[92] button 'Wednesday, January 29, 2025 , 460 US dollars'
			[93] StaticText '$460'
		[94] button 'Thursday, January 30, 2025 , 460 US dollars'
			[95] StaticText '$460'
		[96] button 'Friday, January 31, 2025 , 466 US dollars'
			[97] StaticText '$466'
		[98] StaticText 'February 2025'
		[99] button 'Saturday, February 1, 2025 , 460 US dollars'
			[100] StaticText '$460'
		[101] button 'Sunday, February 2, 2025 , 460 US dollars'
			[102] StaticText '$460'
		[103] button 'Monday, February 3, 2025 , 460 US dollars'
			[104] StaticText '$460'
		[105] button 'Tuesday, February 4, 2025 , 460 US dollars'
			[106] StaticText '$460'
		[107] button 'Wednesday, February 5, 2025 , 460 US dollars'
			[108] StaticText '$460'
		[109] button 'Previous'
		[110] StaticText 'Showing prices in USD for'
		[111] StaticText '7 day trips'
		[112] button 'Done. Search for one-way flights, departing on January 1, 2025' disabled: True
		[113] button 'Search'
	[114] button 'More information on suggested trips from Seattle.' hasPopup: menu
	[115] button 'Explore destinations'
	[116] button 'Las Vegas Mar 25\u2009–\u2009Apr 3 Frontier and Spirit Nonstop 2 hr 33 min 85 US dollars'
	[117] button 'Orlando Mar 24\u2009–\u200930 Frontier 1 stop 30 hr 52 min 286 US dollars'
	[118] button 'Chicago Apr 11\u2009–\u200917 Alaska Nonstop 4 hr 117 US dollars'
	[119] button 'Explore destinations'
	[120] button 'Language\u200b·English (United States)'
	[121] button 'Location\u200b·United States'
	[122] button 'Currency\u200b·USD'
"""

if __name__ == '__main__':
    print (clean_accesibility_tree(a))