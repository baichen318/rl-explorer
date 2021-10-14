# baichen318@gmail.py
# Python2

import argparse
import sys
import json
import re
from xml.etree import ElementTree as ET
from xml.dom import minidom
import copy
import types


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class = argparse.RawDescriptionHelpFormatter,
        description = "Gem5 to McPAT parser")

    parser.add_argument(
        "-d", "--design", type=str, required=True,
        help="input a design name"
    )
    parser.add_argument(
        "--config", "-c", type=str, required=True,
        metavar="PATH",
        help="input config.json from Gem5 output.")
    parser.add_argument(
        "--stats", "-s", type=str, required=True,
        metavar="PATH",
        help="input stats.txt from Gem5 output.")
    parser.add_argument(
        "--template", "-t", type=str, required=True,
        metavar="PATH",
        help="template XML file")
    parser.add_argument(
        "--state",
        nargs='+',
        type=int,
        required=True,
        help="input the state"
    )
    parser.add_argument(
        "--output", "-o", type=argparse.FileType('w'), default="mcpat-in.xml",
        metavar="PATH",
        help="Output file for McPAT input in XML format (default: mcpat-in.xml)")

    return parser


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


class PIParser(ET.XMLTreeBuilder):
   def __init__(self):
       ET.XMLTreeBuilder.__init__(self)
       # assumes ElementTree 1.2.X
       self._parser.CommentHandler = self.handle_comment
       self._parser.ProcessingInstructionHandler = self.handle_pi
       self._target.start("document", {})

   def close(self):
       self._target.end("document")
       return ET.XMLTreeBuilder.close(self)

   def handle_comment(self, data):
       self._target.start(ET.Comment, {})
       self._target.data(data)
       self._target.end(ET.Comment)

   def handle_pi(self, target, data):
       self._target.start(ET.PI, {})
       self._target.data(target + " " + data)
       self._target.end(ET.PI)


def parse_xml(source):
    return ET.parse(source, PIParser())


def read_stats(stats_file):
    global stats
    stats = {}
    with open(stats_file, 'r') as f:
        ignores = re.compile(r'^---|^$')
        stat_line = re.compile(r'([a-zA-Z0-9_\.:-]+)\s+([-+]?[0-9]+\.[0-9]+|[-+]?[0-9]+|nan|inf)')
        count = 0
        for line in f:
            # ignore empty lines and lines starting with "---"
            if not ignores.match(line):
                count += 1
                stat_kind = stat_line.match(line).group(1)
                stat_value = stat_line.match(line).group(2)
                if stat_value == 'nan':
                    print "[WARN]: %s is nan. Setting it to 0." % stat_kind
                    stat_value = '0'
                stats[stat_kind] = stat_value


def read_config(config_file):
    global config
    with open(config_file, 'r') as f:
        config = json.load(f)


def read_mcpat_template(template):
    global mcpat_template
    mcpat_template = parse_xml(template)


def generate_template(output):
    num_cores = len(config["system"]["cpu"])
    private_l2 = config["system"]["cpu"][0].has_key("l2cache")
    shared_l2 = config["system"].has_key("l2")
    if private_l2:
        num_l2 = num_cores
    elif shared_l2:
        num_l2 = 1
    else:
        num_l2 = 0
    elem_counter = 0
    root = mcpat_template.getroot()
    for child in root[0][0]:
        # print(child.attrib.get("name"), child.attrib.get("value"))
        # continue
        # to add elements in correct sequence
        elem_counter += 1
        if child.attrib.get("name") == "number_of_cores":
            child.attrib["value"] = str(num_cores)
        if child.attrib.get("name") == "number_of_L2s":
            child.attrib["value"] = str(num_l2)
        if child.attrib.get("name") == "Private_L2":
            if shared_l2:
                Private_L2 = str(0)
            else:
                Private_L2 = str(1)
            child.attrib["value"] = Private_L2
        temp = child.attrib.get("value")
        # to consider all the cpus in total cycle calculation
        if num_cores > 1 and isinstance(temp, basestring) and "cpu." in temp and temp.split('.')[0] == "stats":
            value = "(" + temp.replace("cpu.", "cpu0.") + ")"
            for i in range(1, num_cores):
                value = value + " + (" + temp.replace("cpu.", "cpu"+str(i)+".") +")"
            child.attrib['value'] = value
        # remove a core template element and replace it with number of cores template elements
        if child.attrib.get("name") == "core":
            core_elem = copy.deepcopy(child)
            core_elem_bak = copy.deepcopy(core_elem)
            for core_counter in range(num_cores):
                core_elem.attrib["name"] = "core" + str(core_counter)
                core_elem.attrib["id"] = "system.core" + str(core_counter)
                for core_child in core_elem:
                    child_id = core_child.attrib.get("id")
                    child_value = core_child.attrib.get("value")
                    child_name = core_child.attrib.get("name")
                    if isinstance(child_name, basestring) and child_name == "x86":
                        if config["system"]["cpu"][core_counter]["isa"][0]["type"] == "X86ISA":
                            child_value = "1"
                        else:
                            child_value = "0"
                    if isinstance(child_id, basestring) and "core" in child_id:
                        # component@core
                        child_id = child_id.replace("core", "core" + str(core_counter))
                    if num_cores > 1 and isinstance(child_value, basestring) and \
                        "cpu." in child_value and "stats" in child_value.split('.')[0]:
                        # stat@core
                        child_value = child_value.replace("cpu.", "cpu" + str(core_counter) + ".")
                    if isinstance(child_value, basestring) and "cpu." in child_value and \
                        "config" in child_value.split('.')[0]:
                        # param@core
                        child_value = child_value.replace("cpu.", "cpu." + str(core_counter) + ".")
                    if len(list(core_child)) != 0:
                        # component@core
                        for core_sub_child in core_child:
                            continue
                            core_sub_child_value = core_sub_child.attrib.get("value")
                            if num_cores > 1 and isinstance(core_sub_child_value, basestring) and \
                                "cpu." in core_sub_child_value and "stats" in core_sub_child_value.split('.')[0]:
                            # stats@component@core
                                core_sub_child_value = core_sub_child_value.replace(
                                    "cpu.",
                                    "cpu" + str(core_counter)+ "."
                                )
                            if isinstance(core_sub_child_value, basestring) and "cpu." in core_sub_child_value and \
                                "config" in core_sub_child_value.split('.')[0]:
                            # config@component@core
                                core_sub_child_value = core_sub_child_value.replace(
                                    "cpu.",
                                    "cpu." + str(core_counter)+ "."
                                )
                            core_sub_child.attrib["value"] = core_sub_child_value
                    if isinstance(child_id, basestring):
                        core_child.attrib["id"] = child_id
                    if isinstance(child_value, basestring):
                        core_child.attrib["value"] = child_value
                root[0][0].insert(elem_counter, core_elem)
                core_elem = copy.deepcopy(core_elem_bak)
                elem_counter += 1
            root[0][0].remove(child)
            elem_counter -= 1
        # remove a L2 template element and replace it with number of L2 template elements
        if child.attrib.get("name") == "L2":
            if private_l2:
                l2_elem = copy.deepcopy(child)
                l2_elem_bak = copy.deepcopy(l2_elem)
                for l2_counter in range(num_l2):
                    l2_elem.attrib["name"] = "L2" + str(l2_counter)
                    l2_elem.attrib["id"] = "system.L2" + str(l2_counter)
                    for l2_child in l2_elem:
                        child_value = l2_child.attrib.get("value")
                        if isinstance(child_value, basestring) and \
                            "cpu." in child_value and "stats" in child_value.split('.')[0]:
                            child_value = child_value.replace("cpu." , "cpu" + str(l2_counter)+ ".")
                        if isinstance(child_value, basestring) and \
                            "cpu." in child_value and "config" in child_value.split('.')[0]:
                            child_value = child_value.replace("cpu." , "cpu." + str(l2_counter)+ ".")
                        if isinstance(child_value, basestring):
                            l2_child.attrib["value"] = child_value
                    root[0][0].insert(elem_counter, l2_elem)
                    l2_elem = copy.deepcopy(l2_elem_bak)
                    elem_counter += 1
                root[0][0].remove(child)
            # else:
            #     child.attrib["name"] = "L20"
            #     child.attrib["id"] = "system.L20"
            #     for l2_child in child:
            #         child_value = l2_child.attrib.get("value")
            #         if isinstance(child_value, basestring) and "cpu.l2cache." in child_value:
            #             child_value = child_value.replace("cpu.l2cache.", "l2.")
    prettify(root)


def get_config(conf):
    split_conf = re.split(r"\.", conf)
    curr_conf = config
    curr_hierarchy = ""
    for x in split_conf:
        curr_hierarchy += x
        if x.isdigit():
            curr_conf = curr_conf[int(x)] 
        elif x in curr_conf:
            curr_conf = curr_conf[x]
    return curr_conf if curr_conf != None else 0


def post_handle_rocket(root):
    for param in root.iter("param"):
        name = param.attrib["name"]
        value = param.attrib["value"]
        if name in [
            "fetch_width", "decode_width", "issue_width",
            "peak_issue_width", "commit_width",
            "decoded_stream_buffer_size", "instruction_window_size",
            "fp_instruction_window_size",
        ]:
            param.attrib["value"] = str(1)
        elif name in ["phy_Regs_IRF_size", "phy_Regs_FRF_size"]:
            param.attrib["value"] = str(32)
        elif name in ["rename_writes", "fp_rename_writes"]:
            param.attrib["value"] = str(0)
    for component in root.iter("component"):
        id = component.attrib["id"]
        name = component.attrib["name"]
        if name == "itlb":
            for param in component.iter("param"):
                name = param.attrib["name"]
                if name == "number_entries":
                    if args.state[1] in [0, 4]:
                        param.attrib["value"] = str(32)
                    elif args.state[1] in [1, 2]:
                        param.attrib["value"] = str(4)
                    else:
                        assert args.state[1] in [3, 5]
                        param.attrib["value"] = str(16)
        if name == "icache":
            for param in component.iter("param"):
                name = param.attrib["name"]
                if name == "icache_config":
                    icache_size = [
                        4 * 64 * 64,
                        1 * 64 * 64,
                        2 * 64 * 64,
                        2 * 64 * 64,
                        2 * 64 * 64,
                        1 * 64 * 64
                    ]
                    icache_assoc = [4, 1, 2, 2, 2, 1]
                    param.attrib["value"] = "%s,%s,%s,%s,%s,%s,%s,%s" % (
                        icache_size[args.state[1]],
                        64,
                        icache_assoc[args.state[1]],
                        1,
                        1,
                        3,
                        64,
                        1
                    )
                if name == "buffer_sizes":
                    param.attrib["value"] = "%s,%s,%s,%s" % (
                        2, 2, 2, 2
                    )
        if name == "dtlb":
            for param in component.iter("param"):
                name = param.attrib["name"]
                if name == "number_entries":
                    if args.state[5] in [0, 5, 7, 9]:
                        param.attrib["value"] = str(32)
                    elif args.state[5] in [1, 2, 3, 8]:
                        param.attrib["value"] = str(4)
                    else:
                        assert args.state[5] in [4, 6]
                        param.attrib["value"] = str(16)
        if name == "dcache":
            for param in component.iter("param"):
                name = param.attrib["name"]
                if name == "dcache_config":
                    dcache_size = [
                        64 * 4 * 64,
                        64 * 1 * 64,
                        64 * 1 * 64,
                        64 * 2 * 64,
                        64 * 2 * 64,
                        64 * 2 * 64,
                        64 * 1 * 64,
                        64 * 4 * 64,
                        64 * 1 * 64,
                        64 * 4 * 64,
                    ]
                    dcache_assoc = [4, 1, 1, 2, 2, 2, 1, 4, 1, 4]
                    param.attrib["value"] = "%s,%s,%s,%s,%s,%s,%s,%s" % (
                        dcache_size[args.state[5]],
                        64,
                        dcache_assoc[args.state[5]],
                        1,
                        1,
                        3,
                        64,
                        1
                    )
                if name == "buffer_sizes":
                    mshr = [1, 0, 0, 0, 0, 1, 0, 2, 2, 2]
                    param.attrib["value"] = "%s,%s,%s,%s" % (
                        2 if mshr[args.state[5]] == 0 else mshr[args.state[5]] << 1,
                        2, 2, 2
                    )
            if name == "BTB":
                for param in component.iter("param"):
                    name = param.attrib["name"]
                    if name == "BTB_config":
                        capacity = [0, 512, 256, 1024, 512]
                        param.attrib["value"] = "%s,%s,%s,%s,%s,%s" % (
                            capacity[args.state[0]], 2, 1, 1, 1
                        )
        if name == "L2":
            for param in component.iter("param"):
                name = param.attrib["name"]
                if name == "L2_config":
                    param.attrib["value"] = "%s,%s,%s,%s,%s,%s,%s,%s" % (
                        67108864,
                        32,
                        4,
                        8,
                        8,
                        23,
                        32,
                        1
                    )
                if name == "buffer_sizes":
                    param.attrib["value"] = "%s,%s,%s,%s" % (
                        16, 16, 16, 16
                    )
                if name == "clockrate":
                    param.attrib["value"] = "666.667"
                if name == "vdd":
                    param.attrib["value"] = "1.5"


def post_handle_boom(root):
    for param in root.iter("param"):
        name = param.attrib["name"]
        value = param.attrib["value"]
        if name == "fp_issue_width":
            fp_issue_width = [8, 16, 24, 32, 12, 20, 28]
            param.attrib["value"] = str(fp_issue_width[args.state[7]])
        if name in [
            "fp_issue_width"
        ]:
            param.attrib["value"] = str(1)
        elif name in ["phy_Regs_IRF_size", "phy_Regs_FRF_size"]:
            param.attrib["value"] = str(32)
        elif name in ["rename_writes", "fp_rename_writes"]:
            param.attrib["value"] = str(0)
    for component in root.iter("component"):
        id = component.attrib["name"]
        name = component.attrib["name"]
        if name == "itlb":
            for param in component.iter("param"):
                name = param.attrib["name"]
                if name == "number_entries":
                    param.attrib["value"] = str(32)
        if name == "icache":
            for param in component.iter("param"):
                name = param.attrib["name"]
                if name == "icache_config":
                    param.attrib["value"] = "%s,%s,%s,%s,%s,%s,%s,%s" % (
                        int(args.state[1]) * 64 * 64,
                        64,
                        args.state[1],
                        1,
                        1,
                        3,
                        64,
                        1
                    )
                if name == "buffer_sizes":
                    param.attrib["value"] = "%s,%s,%s,%s" % (
                        2, 2, 2, 2
                    )
        if name == "dtlb":
            for param in component.iter("param"):
                name = param.attrib["name"]
                if name == "number_entries":
                    param.attrib["value"] = str(32)
        if name == "dcache":
            for param in component.iter("param"):
                name = param.attrib["name"]
                if name == "dcache_config":
                    param.attrib["value"] = "%s,%s,%s,%s,%s,%s,%s,%s" % (
                        int(args.state[1]) * 64 * 64,
                        64,
                        args.state[1],
                        1,
                        1,
                        3,
                        64,
                        1
                    )
                if name == "buffer_sizes":
                    mshr = [2, 4 ,8, 16, 4]
                    param.attrib["value"] = "%s,%s,%s,%s" % (
                        mshr[args.state[9]],
                        mshr[args.state[9]],
                        mshr[args.state[9]],
                        mshr[args.state[9]]
                    )


def post_handle(root):
    if args.design == "rocket":
        post_handle_rocket(root)
    elif args.design == "boom":
        post_handle_boom(root)
    else:
        assert args.design == "cva6", \
            "[ERROR]: unsupported design: %s" % args.design


def write_mcpat_xml(output_path):
    root = mcpat_template.getroot()
    # boom_root = boomParams.getroot()
    pattern = re.compile(r"config\.([][a-zA-Z0-9_:\.]+)")
    # replace params with values from the GEM5 config file
    for param in root.iter("param"):
        name = param.attrib["name"]
        value = param.attrib["value"]
        if "config" in value:
            # param
            all_confs = pattern.findall(value)
            for conf in all_confs:
                conf_value = get_config(conf)
                if type(conf_value) == dict or type(conf_value) == list:
                    conf_value = 0
                    print("[WARN]: %s does not exist in gem5 config." % conf)
                value = re.sub("config."+ conf, str(conf_value), value)
            if "," in value:
                # e.g., pipelines_per_core, pipeline_depth
                exprs = re.split(',', value)
                for i in range(len(exprs)):
                    exprs[i] = str(eval(exprs[i]))
                param.attrib["value"] = ','.join(exprs)
            else:
                param.attrib["value"] = str(eval(str(value)))
    # replace stats with values from the GEM5 stats file
    stat_pattern = re.compile(r"stats\.([a-zA-Z0-9_:\.]+)")
    for stat in root.iter("stat"):
        name = stat.attrib["name"]
        value = stat.attrib["value"]
        if "stats" in value:
            all_stats = stat_pattern.findall(value)
            expr = value
            for i in range(len(all_stats)):
                if all_stats[i] in stats:
                    expr = re.sub("stats.%s" % all_stats[i], stats[all_stats[i]], expr)
                else:
                    expr = re.sub('stats.%s' % all_stats[i], str(0), expr)
                    print("[WARN]: %s does not exist in gem5 stats." % all_stats[i])
            if "config" not in expr and "stats" not in expr:
                try:
                    stat.attrib["value"] = str(eval(expr))
                except ZeroDivisionError as e:
                    print("[ERROR]: %s" % e)

    post_handle(root)

    # Write out the xml file
    mcpat_template.write(output_path)


def main():
    read_stats(args.stats)
    read_config(args.config)
    read_mcpat_template(args.template)
    generate_template(args.output)
    write_mcpat_xml(args.output)


if __name__ == '__main__':
    args = create_parser().parse_args()
    main()
