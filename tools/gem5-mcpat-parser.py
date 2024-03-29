# Author: baichen318@gmail.com


import os
import re
import sys
import yaml
import json
import copy
import types
import argparse
from xml.dom import minidom
from xml.etree import ElementTree as ET
from utils.utils import get_configs, round_power_of_two, \
    info, warn, error, load_excel


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class = argparse.RawDescriptionHelpFormatter,
        description = "Gem5 to McPAT parser")

    parser.add_argument(
        "-y",
        "--yaml",
        type=str,
        required=True,
        metavar="PATH",
        help="input a yaml config."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        metavar="PATH",
        help="input config.json from Gem5 output."
    )
    parser.add_argument(
        "-s",
        "--stats",
        type=str,
        required=True,
        metavar="PATH",
        help="input stats.txt from Gem5 output."
    )
    parser.add_argument(
        "-t",
        "--template",
        type=str, required=True,
        metavar="PATH",
        help="template XML file"
    )
    parser.add_argument(
        "-x",
        "--state",
        nargs='+',
        type=int,
        required=True,
        help="input the state"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="mcpat-in.xml",
        metavar="PATH",
        help="Output file for McPAT input in XML format (default: mcpat-in.xml)"
    )

    return parser


def parse_xml(source):
    return ET.parse(source)


def read_stats(stats_file):
    global stats
    with open(stats_file, 'r') as f:
        ignores = re.compile(r"^---|^$")
        stat_line = re.compile(r"([a-zA-Z0-9_\.:-]+)\s+([-+]?[0-9]+\.[0-9]+|[-+]?[0-9]+|nan|inf)")
        count = 0
        for line in f:
            # ignore empty lines and lines starting with "---"
            if not ignores.match(line):
                count += 1
                try:
                    stat_kind = stat_line.match(line).group(1)
                    stat_value = stat_line.match(line).group(2)
                except AttributeError as e:
                    continue
                if stat_value == "nan":
                    warn("%s is \"nan\". set it to 0." % stat_kind)
                    stat_value = '0'
                stats[stat_kind] = stat_value


def read_config(config_file):
    global config
    with open(config_file, 'r') as f:
        config = json.load(f)


def read_mcpat_template(template):
    global mcpat_template
    mcpat_template = parse_xml(template)


def get_num_cores():
    return len(config["system"]["cpu"])


def get_private_l2():
    return "l2cache" in config["system"]["cpu"][0]


def get_shared_l2():
    return "l2" in config["system"]


def get_num_of_l2s():
    if get_private_l2():
        return get_num_cores()
    # NOTICE: McPAT assumes that a core must has a L2C
    # return '1' if get_shared_l2() else '0'
    return '1'


def get_private_l2_flag():
    if get_shared_l2():
        return '0'
    else:
        if get_private_l2():
            return '1'
        else:
            return '0'


def if_cpu_stats_from_template(child):
    value = child.attrib.get("value")
    if value is not None and \
        "cpu." in value and \
        value.split('.')[0] == "stats":
        return child
    return None


def if_cpu_config_from_template(child):
    value = child.attrib.get("value")
    if value is not None and \
        "cpu." in value and \
        "config" in value.split('.')[0]:
        return child
    return None


def get_isa(idx):
    return '1' \
        if config["system"]["cpu"][idx]["isa"][0]["type"] == "X86ISA" else '0'


def extend_to_all_cpu(core):
    # it is tricky and clumpy
    # it cannot handle multi-core cofigurations
    num_cores = get_num_cores()
    for idx in range(num_cores):
        core.attrib["name"] = "core{}".format(idx)
        core.attrib["id"] = "system.core{}".format(idx)
        for child in core:
            _id = child.attrib.get("id")
            name = child.attrib.get("name")
            value = child.attrib.get("value")
            if name == "x86":
                value = get_isa(idx)
            if _id and "core" in _id:
                _id = _id.replace(
                    "core",
                    "core{}".format(idx)
                )
            if num_cores > 1 and if_cpu_stats_from_template(child) is not None:
                value = value.replace("cpu.", "cpu{}.".format(idx))
            if if_cpu_config_from_template(child) is not None:
                value = value.replace("cpu.", "cpu.{}.".format(idx))
            if len(list(child)) != 0:
                for _child in child:
                    _value = _child.attrib.get("value")
                    if num_cores > 1 and if_cpu_stats_from_template(_child) is not None:
                        _value = _value.replace(
                            "cpu.",
                            "cpu{}.".format(idx)
                        )
                    if num_cores > 1 and if_cpu_config_from_template(_child) is not None:
                        _value = _value.replace(
                            "cpu.",
                            "cpu.{}.".format(idx)
                        )
                    _child.attrib["value"] = _value
            if _id:
                child.attrib["id"] = _id
            if value:
                child.attrib["value"] = value


def extend_to_all_l2(l2):
    # it is tricky and clumpy
    # it cannot handle multi-core cofigurations
    num_cores = get_num_cores()
    for idx in range(num_cores):
        l2.attrib["id"] = "system.L2{}".format(idx)
        l2.attrib["name"] = "L2{}".format(idx)
        for child in l2:
            value = child.attrib.get("value")
            if if_cpu_stats_from_template(child) is not None:
                value = value.replace(
                    "cpu.",
                    "cpu{}.".format(idx)
                )
            if if_cpu_config_from_template(child) is not None:
                value = value.replace(
                    "cpu.",
                    "cpu.{}.".format(idx)
                )
            if value:
                child.attrib["value"] = value


def extend_misc_components_to_all_cpu(child):
    num_cores = get_num_cores()
    if num_cores > 1:
        child = if_cpu_stats_from_template(child)
        if child is not None:
            value = child.attrib.get("value")
            value = "({})".format(
                value.replace("cpu.", "cpu0.")
            )
            for i in range(1, num_cores):
                value = "{} + ({})".format(
                    value,
                    value.replace("cpu.", "cpu{}.".format(i))
                )
            child.attrib["value"] = value


def generate_mcpat_xml():
    def format_xml(root):
        rough_string = ET.tostring(root, "utf-8")
        reparsed = minidom.parseString(rough_string)
        reparsed.toprettyxml(indent='\t')

    root = mcpat_template.getroot()
    num_cores = get_num_cores()
    for child in root[0]:
        name = child.attrib.get("name")
        if name == "number_of_cores":
            child.attrib["value"] = str(num_cores)
        if name == "number_of_L2s":
            child.attrib["value"] = get_num_of_l2s()
        if name == "Private_L2":
            child.attrib["value"] = get_private_l2_flag()
        if name == "core":
            extend_to_all_cpu(child)
        if name == "L2":
            if get_private_l2_flag():
                extend_to_all_l2(child)
        extend_misc_components_to_all_cpu(child)
    format_xml(root)


def get_config(conf):
    split_conf = re.split(r"\.", conf)
    curr_conf = config
    for x in split_conf:
        if x.isdigit():
            curr_conf = curr_conf[int(x)]
        elif x in curr_conf:
            curr_conf = curr_conf[x]
    return curr_conf if curr_conf != None else 0


def post_handle_rocket(root):
    from dse.env.rocket.design_space import parse_design_space

    design_space = parse_design_space(configs)
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
                    itlb = args.state[4]
                    param.attrib["value"] = str(itlb)
        if name == "icache":
            for param in component.iter("param"):
                name = param.attrib["name"]
                if name == "icache_config":
                    param.attrib["value"] = "%s,%s,%s,%s,%s,%s,%s,%s" % (
                        args.state[3] * 64 * 64,
                        64,
                        args.state[3],
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
                    param.attrib["value"] = str(
                        args.state[12]
                    )
        if name == "dcache":
            for param in component.iter("param"):
                name = param.attrib["name"]
                if name == "dcache_config":
                    param.attrib["value"] = "%s,%s,%s,%s,%s,%s,%s,%s" % (
                        args.state[10] * args.state[11] * 64,
                        64,
                        args.state[11],
                        1,
                        1,
                        3,
                        64,
                        1
                    )
                if name == "buffer_sizes":
                    mshr = args.state[13]
                    param.attrib["value"] = "%s,%s,%s,%s" % (
                        2 if mshr == 0 else mshr << 1, 2, 2, 2
                    )
            if name == "BTB":
                for param in component.iter("param"):
                    name = param.attrib["name"]
                    if name == "BTB_config":
                        bht = args.state[2]
                        param.attrib["value"] = "%s,%s,%s,%s,%s,%s" % (
                            bht, 2, 1, 1, 1
                        )
        if name == "L20":
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
    from dse.env.boom.design_space import parse_design_space

    design_space = parse_design_space(configs)
    for param in root.iter("param"):
        name = param.attrib["name"]
        value = param.attrib["value"]
        if name == "fp_issue_width":
            param.attrib["value"] = str(args.state[15])
        if name in [
            "fp_issue_width"
        ]:
            param.attrib["value"] = str(args.state[15])
        elif name in ["phy_Regs_IRF_size"]:
            param.attrib["value"] = str(args.state[7])
        elif name in ["phy_Regs_FRF_size"]:
            param.attrib["value"] = str(args.state[8])
        elif name in ["rename_writes", "fp_rename_writes"]:
            param.attrib["value"] = str(0)
        elif name in ["instruction_buffer_size"]:
            param.attrib["value"] = str(round_power_of_two(int(value)))
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
                        args.state[20] * args.state[21] * 64,
                        64,
                        args.state[20],
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
                    param.attrib["value"] = str(args.state[22])
        if name == "dcache":
            for param in component.iter("param"):
                name = param.attrib["name"]
                if name == "dcache_config":
                    param.attrib["value"] = "%s,%s,%s,%s,%s,%s,%s,%s" % (
                        args.state[22] * args.state[23] * 64,
                        64,
                        args.state[22],
                        1,
                        1,
                        3,
                        64,
                        1
                    )
                if name == "buffer_sizes":
                    mshr = args.state[24]
                    param.attrib["value"] = "%s,%s,%s,%s" % (
                        mshr,
                        mshr,
                        mshr,
                        mshr
                    )
        if name == "L20":
            for param in component.iter("param"):
                name = param.attrib["name"]
                if name == "L2_config":
                    param.attrib["value"] = "%s,%s,%s,%s,%s,%s,%s,%s" % (
                        134217728, 64, 8, 1, 1, 5, 64, 1
                    )
                if name == "buffer_sizes":
                    param.attrib["value"] = "8,8,8,8"


def post_handle(root):
    if "BOOM" in configs["algo"]["design"]:
        post_handle_boom(root)
    else:
        assert configs["algo"]["design"] == "Rocket", \
            "[ERROR]: unsupported design: {}".format(args.design)
        post_handle_rocket(root)


def write_mcpat_xml(output_path):
    root = mcpat_template.getroot()
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
                    warn("%s does not exist in gem5 config." % conf)
                value = re.sub("config." + conf, str(conf_value), value)
            if ',' in value:
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
                    warn("%s does not exist in gem5 stats." % all_stats[i])
            if "config" not in expr and "stats" not in expr:
                try:
                    stat.attrib["value"] = str(eval(expr))
                except ZeroDivisionError as e:
                    error("{}".format(e))

    post_handle(root)

    # Write out the xml file
    with open(output_path, 'wb') as f:
        mcpat_template.write(f)
    info("create McPAT input xml: {}.".format(output_path))


def main():
    read_stats(args.stats)
    read_config(args.config)
    read_mcpat_template(args.template)
    generate_mcpat_xml()
    write_mcpat_xml(args.output)


if __name__ == '__main__':
    args = create_parser().parse_args()
    configs = get_configs(args.yaml)
    stats = {}
    config = None
    mcpat_template = None
    main()
