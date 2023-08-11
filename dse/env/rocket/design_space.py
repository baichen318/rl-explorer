# Author: baichen318@gmail.com


import os
import numpy as np
from typing import List
from collections import OrderedDict
from dse.env.base_design_space import DesignSpace, Macros
from utils.utils import load_excel, assert_error, if_exist


class RocketMacros(Macros):
    def __init__(self, root):
        super(RocketMacros, self).__init__()
        self.macros["chipyard-research-root"] = root
        self.macros["workstation-root"] = os.path.join(
            self.macros["chipyard-research-root"],
            "workstation"
        )
        self.macros["core-cfg"] = os.path.join(
            self.macros["chipyard-research-root"],
            "generators",
            "rocket-chip",
            "src",
            "main",
            "scala",
            "subsystem",
            "Configs.scala"
        )
        self.macros["soc-cfg"] = os.path.join(
            self.macros["chipyard-research-root"],
            "generators",
            "chipyard",
            "src",
            "main",
            "scala",
            "config",
            "RocketConfigs.scala"
        )
        self.validate_macros()

    def validate_macros(self):
        if_exist(self.macros["core-cfg"], strict=True)
        if_exist(self.macros["soc-cfg"], strict=True)

    def get_mapping_params(self, vec, idx):
        return self.components_mappings[self.components[idx]][vec[idx]]

    def generate_mul_div(self, vec, idx):
        params = self.get_mapping_params(vec, idx)
        return """Some(MulDivParams(
                    mulUnroll = %d,
                    mulEarlyOut = %s,
                    divEarlyOut = %s
                ))""" % (
              params[0],
              params[1],
              params[2]
            )

    def generate_fpu(self, vec, idx):
        params = self.get_mapping_params(vec, idx)[0]
        if params == 0:
            return "None"
        else:
            return "Some(FPUParams())"

    def generate_use_vm(self, vec, idx):
        return self.get_mapping_params(vec, idx)[0]

    def generate_btb(self, vec, idx):
        params =  self.get_mapping_params(vec, idx)
        if params[0] == 0:
            return "None"
        else:
            return """Some(BTBParams(
                    nRAS = %d,
                    nEntries = %d,
                    bhtParams = Some(BHTParams(nEntries=%d))
                )
            )""" % (
                  params[0],
                  params[1],
                  params[2]
              )

    def generate_dcache_and_mmu(self, vec, idx):
        params = self.get_mapping_params(vec, idx)
        return """Some(DCacheParams(
                    rowBits = site(SystemBusKey).beatBits,
                    nSets = %d,
                    nWays = %d,
                    nTLBSets = 1,
                    nTLBWays=%d,
                    nMSHRs=%d,
                    blockBytes = site(CacheBlockBytes)
                )
            )""" % (
                params[0],
                params[1],
                params[2],
                params[3]
            )

    def generate_icache_and_mmu(self, vec, idx):
        params = self.get_mapping_params(vec, idx)
        return """Some(ICacheParams(
                    rowBits = site(SystemBusKey).beatBits,
                    nSets = 64,
                    nWays = %d,
                    nTLBSets = 1,
                    nTLBWays = %d,
                    blockBytes = site(CacheBlockBytes)
                )
            )""" % (
                params[0],
                params[1]
            )

    def generate_core_cfg_impl(self, name, vec):
        codes = '''
class %s(n: Int, overrideIdOffset: Option[Int] = None) extends Config((site, here, up) =>
{
  case RocketTilesKey => {
    val prev = up(RocketTilesKey, site)
    val idOffset = overrideIdOffset.getOrElse(prev.size)
    val rocket_core = RocketTileParams(
      core = RocketCoreParams(
        mulDiv = %s,
        fpu = %s,
        useVM = %s,
      ),
      btb = %s,
      dcache = %s,
      icache = %s
    )
    List.tabulate(n)(i => rocket_core.copy(hartId = i + idOffset)) ++ prev
  }
})
''' % (
        name,
        self.generate_mul_div(vec, 3),
        self.generate_fpu(vec, 2),
        self.generate_use_vm(vec, 4),
        self.generate_btb(vec, 0),
        self.generate_dcache_and_mmu(vec, 5),
        self.generate_icache_and_mmu(vec, 1)
)
        return codes

    def write_core_cfg_impl(self, codes):
        with open(self.macros["core-cfg"], 'a') as f:
            f.writelines(codes)

    def generate_soc_cfg_impl(self, soc_name, core_name):
        codes = '''
class %s extends Config(
  new freechips.rocketchip.subsystem.%s(1) ++
  new chipyard.config.AbstractConfig)
''' % (
        soc_name,
        core_name
    )
        return codes

    def write_soc_cfg_impl(self, codes):
        with open(self.macros["soc-cfg"], 'a') as f:
            f.writelines(codes)

    def vec_to_embedding(self, vec):
        embedding = []
        for idx in range(len(vec)):
            for v in self.get_mapping_params(vec, idx):
                if v == "true":
                    v = 1
                elif v == "false":
                    v = 0
                embedding.append(v)
        return embedding

    def embedding_to_vec(self, embedding):
        vec = []
        c, idx = 0, 0
        while idx < len(embedding):
            prev_idx = idx
            num = len(self.components_mappings[self.components[c]]["description"])
            feature = embedding[idx : idx + num]
            for k, v in self.components_mappings[self.components[c]].items():
                if "true" in v or "false" in v:
                    for i in range(len(v)):
                        if v[i] == "true":
                            v[i] = 1
                        elif v[i] == "false":
                            v[i] = 0
                if v == feature:
                    vec.append(int(k))
                    idx += num
                    break
            if prev_idx == idx:
                break
            c += 1
        assert len(vec) == self.dims, assert_error(
          "invalid vec: {}".format(vec)
        )
        return vec


class RocketDesignSpace(DesignSpace, RocketMacros):
    def __init__(self, root, descriptions, components_mappings, size):
        """
            descriptions: <class "collections.OrderedDict">
            Example:
            descriptions = {
                "Rocket": {
                    "BTB": [1, 2, 3, 4, 5],
                    "R. I-Cache": [1, 2, 3, 4, 5, 6],
                    "FPU": [1, 2],
                    "mulDiv": [1, 2, 3],
                    "useVM": [1, 2],
                    "R. D-Cache": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                }
            }

            components_mappings: <class "collections.OrderedDict">
            Example:
            components_mappings = {
                "BTB": {
                    "description": [
                        "nRAS", "BTBParams.nEntries", "BHTParams.nEntries"
                    ],
                    "1": [0, 0, 0],
                    "2": [6, 28, 512],
                    "3": [3, 14, 256],
                    "4": [12, 56, 1024],
                    "5": [8, 32, 512]
                },
                "IFU": {
                    "R. I-Cache     ": ["nWays", "nTLBWays"]
                    "1": [4, 32],
                    "2": [1, 4],
                    "3": [2, 4],
                    "4": [2, 16],
                    "5": [2, 32],
                    "6": [1, 16]
                }
            }

            size: <int> the size of the entire design space
        """
        self.descriptions = descriptions
        self.components_mappings = components_mappings
        # construct look-up tables
        # self.designs: <list>
        # Example:
        #   ["Rocket"]
        self.designs = list(self.descriptions.keys())
        # self.components: <list>
        # Example:
        #   ["BTB", "R. I-Cache", "FPU", "mulDiv", "useVM", "R. D-Cache"]
        self.components = list(self.descriptions[self.designs[0]].keys())
        # Example:
        #       length of ["branchPredictor", "fetchWidth", "numFetchBufferEntries", ...]
        self.embedding_dims = self.construct_embedding_dims()
        self.design_size = self.construct_design_size()
        # self.acc_design_size: <list> list of accumulated sizes of each design
        # Example:
        #   self.design_size = [a, b, c, d, e] # accordingly
        #   self.acc_design_size = [a, a + b, a + b + c, a + b + c + d, a + b + c + d + e]
        self.acc_design_size = list(map(
                lambda x, idx: np.sum(x[:idx]),
                [self.design_size for i in range(len(self.design_size))],
                range(1, len(self.design_size) + 1)
            )
        )
        self.component_dims = self.construct_component_dims()
        DesignSpace.__init__(self, size, len(self.components))
        RocketMacros.__init__(self, root)

    def construct_embedding_dims(self):
        dims = 0
        for k, v in self.descriptions[self.designs[0]].items():
            dims += len(self.components_mappings[k]["description"])
        return dims

    def construct_design_size(self):
        """
            design_size: <list> list of sizes of each design
            Example:
                [a, b, c, d, e] and np.sum([[a, b, c, d, e]]) = self.size
        """
        design_size = []
        for k, v in self.descriptions.items():
            _design_size = []
            for _k, _v in v.items():
                _design_size.append(len(_v))
            design_size.append(np.prod(_design_size))
        return design_size

    def construct_component_dims(self):
        """
            component_dims: <list> list of dimensions of each
                                   component w.r.t. each design
            Example:
                [
                    [a, b, c, d, e], # => dimensions of components of the 1st design
                    [f, g, h, i, j]  # => dimensions of components of the 2nd design
                ]
        """
        component_dims = []
        for k, v in self.descriptions.items():
            _component_dims = []
            for _k, _v in v.items():
                _component_dims.append(len(_v))
            component_dims.append(_component_dims)
        return component_dims

    def idx_to_vec(self, idx):
        """
            idx: <int> the index of a microarchitecture
        """
        idx -= 1
        assert idx >= 0, "[ERROR]: invalid index."
        assert idx < self.size, "[ERROR]: index exceeds the search space."
        vec = []
        design = np.where(np.array(self.acc_design_size) > idx)[0][0]
        if design >= 1:
            # NOTICE: subtract the offset
            idx -= self.acc_design_size[design - 1]
        for dim in self.component_dims[design]:
            vec.append(idx % dim)
            idx //= dim
        for i in range(len(vec)):
            vec[i] = self.descriptions[self.designs[design]][self.components[i]][vec[i]]
        return vec

    def vec_to_idx(self, vec):
        """
            vec: <list> the list of a microarchitecture encoding
        """

        idx = 0
        design = 0
        # reset "fetchWidth" and "decodeWidth"
        for i in range(len(vec)):
            vec[i] = self.descriptions[self.designs[design]][self.components[i]].index(vec[i])
        for j, k in enumerate(vec):
            idx += int(np.prod(self.component_dims[design][:j])) * k
        if design >= 1:
            # NOTICE: add the offset
            idx += self.acc_design_size[design - 1]
        assert idx >= 0, "[ERROR]: invalid index."
        assert idx < self.size, "[ERROR]: index exceeds the search space."
        idx += 1
        return idx

    def idx_to_embedding(self, idx):
        vec = self.idx_to_vec(idx)
        return self.vec_to_embedding(vec)

    def embedding_to_idx(self, embedding):
        vec = self.embedding_to_vec(embedding)
        return self.vec_to_idx(vec)

    def generate_core_cfg(self, batch):
        """
            generate core configurations
        """
        codes = []
        for idx in batch:
            codes.append(self.generate_core_cfg_impl(
                    "WithN{}Rocket".format(idx),
                    self.idx_to_vec(idx)
                )
            )
        return codes

    def write_core_cfg(self, codes):
        self.write_core_cfg_impl(codes)

    def generate_soc_cfg(self, batch):
        """
            generate soc configurations
        """
        codes = []
        for idx in batch:
            codes.append(self.generate_soc_cfg_impl(
                    "Rocket{}Config".format(idx),
                    "WithN{}Rocket".format(idx)
                )
            )
        return codes

    def generate_chisel_codes(self, batch):
        codes = self.generate_core_cfg(batch)
        self.write_core_cfg(codes)
        codes = self.generate_soc_cfg(batch)
        self.write_soc_cfg(codes)

    def write_soc_cfg(self, codes):
        self.write_soc_cfg_impl(codes)


def parse_design_space_sheet(design_space_sheet):
    descriptions = OrderedDict()
    head = design_space_sheet.columns.tolist()

    # parse design space
    for row in design_space_sheet.values:
        # extract designs
        descriptions[row[0]] = OrderedDict()
        # extract components
        for col in range(1, len(head) - 1):
            descriptions[row[0]][head[col]] = []
        # extract candidate values
        for col in range(1, len(head) - 1):
            try:
                # multiple candidate values
                for item in list(map(lambda x: int(x), row[col].split(','))):
                    descriptions[row[0]][head[col]].append(item)
            except AttributeError:
                # single candidate value
                descriptions[row[0]][head[col]].append(row[col])
    return descriptions



def parse_components_sheet(components_sheet):
    components_mappings = OrderedDict()
    head = components_sheet.columns.tolist()

    # construct look-up tables
    # mappings: <list> [name, width, idx]
    # Example:
    #   mappings = [("ISU", 10, 0), ("IFU", 4, 10), ("ROB", 2, 14)]
    mappings = []
    for i in range(len(head)):
        if not head[i].startswith("Unnamed"):
            if i == 0:
                name, width, idx = head[i], 1, i
            else:
                mappings.append((name, width, idx))
                name, width, idx = head[i], 1, i
        else:
            width += 1
    mappings.append((name, width, idx))

    for name, width, idx in mappings:
        # extract components
        components_mappings[name] = OrderedDict()
        # extract descriptions
        components_mappings[name]["description"] = []
        for i in range(idx + 1, idx + width):
            components_mappings[name]["description"].append(components_sheet.values[0][i])
        # extract candidate values
        # get number of rows, a trick to test <class "float"> of nan
        nrow = np.where(components_sheet[name].values == \
            components_sheet[name].values)[0][-1]
        for i in range(1, nrow + 1):
            components_mappings[name][int(i)] = \
                list(components_sheet.values[i][idx + 1: idx + width])
    return components_mappings


def parse_rocket_design_space(root, design_space_sheet, components_sheet):
    """
        design_space_sheet: <class "pandas.core.frame.DataFrame">
        components_sheet: <class "pandas.core.frame.DataFrame">
    """
    # parse design space
    descriptions = parse_design_space_sheet(design_space_sheet)

    # parse components
    components_mappings = parse_components_sheet(components_sheet)

    return RocketDesignSpace(
        root,
        descriptions,
        components_mappings,
        int(design_space_sheet.values[0][-1])
    )

def parse_design_space(configs):
    """
        configs: <dict>
    """
    design_space_excel = os.path.abspath(
        os.path.join(
            configs["env"]["vlsi"]["chipyard-research-root"],
            "workstation",
            "configs",
            "design-space",
            "design-space-v2.xlsx"
        )
    )
    rocket_design_space_sheet = load_excel(
        design_space_excel,
        sheet_name="Rocket Design Space"
    )
    components_sheet = load_excel(design_space_excel, sheet_name="Components")
    rocket_design_space = parse_rocket_design_space(
        configs["env"]["vlsi"]["chipyard-research-root"],
        rocket_design_space_sheet,
        components_sheet
    )
    return rocket_design_space
