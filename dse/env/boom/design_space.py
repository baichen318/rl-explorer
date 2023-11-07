

import os
import numpy as np
from typing import List
from collections import OrderedDict
from dse.env.base_design_space import DesignSpace, Macros
from utils.utils import load_excel, assert_error, if_exist


class BOOMMacros(Macros):
  def __init__(self, root):
    super(BOOMMacros, self).__init__()
    self.macros["chipyard-research-root"] = root
    self.macros["workstation-root"] = os.path.join(
      self.macros["chipyard-research-root"],
      "workstation"
    )
    self.macros["core-cfg"] = os.path.join(
      self.macros["chipyard-research-root"],
      "generators",
      "boom",
      "src",
      "main",
      "scala",
      "common",
      "config-mixins.scala"
    )
    self.macros["soc-cfg"] = os.path.join(
      self.macros["chipyard-research-root"],
        "generators",
      "chipyard",
      "src",
      "main",
      "scala",
      "config",
      "BoomConfigs.scala"
    )
    self.validate_macros()

  def validate_macros(self):
    if_exist(self.macros["core-cfg"], strict=True)
    if_exist(self.macros["soc-cfg"], strict=True)

  def get_mapping_params(self, vec, idx):
    return self.components_mappings[self.components[idx]][vec[idx]]

  def generate_branch_predictor(self, vec, idx):
    bp = self.get_mapping_params(vec, idx)[0]
    if bp == "TAGEL":
      return "new WithTAGELBPD ++"
    elif bp == "Gshare":
      return "new WithBoom2BPD ++"
    else:
      return "new WithAlpha21264BPD ++"

  def generate_fetch_width(self, vec, idx):
    return self.get_mapping_params(vec, idx)[0]

  def generate_fetch_buffer(self, vec, idx):
    return self.get_mapping_params(vec, idx)[0]

  def generate_ftq(self, vec, idx):
    return self.get_mapping_params(vec, idx)[1]

  def generate_max_br_count(self, vec, idx):
    return self.get_mapping_params(vec, idx)[0]

  def generate_decode_width(self, vec, idx):
    return self.get_mapping_params(vec, idx)[0]

  def generate_rob_entries(self, vec, idx):
    return self.get_mapping_params(vec, idx)[0]

  def generate_issue_parames(self, vec, idx):
    params = self.get_mapping_params(vec, idx)
    return """Seq(
                IssueParams(issueWidth=%d, numEntries=%d, iqType=IQT_MEM.litValue, dispatchWidth=%d),
                IssueParams(issueWidth=%d, numEntries=%d, iqType=IQT_INT.litValue, dispatchWidth=%d),
                IssueParams(issueWidth=%d, numEntries=%d, iqType=IQT_FP.litValue, dispatchWidth=%d)
              )""" % (
            # IQT_MEM.issueWidth, IQT_MEM.numEntries, IQT_MEM.dispatchWidth
            params[0], params[1], params[2],
            # IQT_INT.issueWidth, IQT_INT.numEntries, IQT_INT.dispatchWidth
            params[3], params[4], params[5],
            # IQT_FP.issueWidth, IQT_FP.numEntries, IQT_FP.dispatchWidth
            params[6], params[7], params[8]
        )

  def generate_phy_registers(self, vec, idx):
    params = self.get_mapping_params(vec, idx)
    return """numIntPhysRegisters = %d,
              numFpPhysRegisters = %d""" % (
            params[0],
            params[1]
        )

  def generate_lsu(self, vec, idx):
    params = self.get_mapping_params(vec, idx)
    return """numLdqEntries = %d,
              numStqEntries = %d""" % (
            params[0],
            params[1]
        )

  def generate_icache(self, vec, idx):
    params = self.get_mapping_params(vec, idx)
    return """Some(
              ICacheParams(
                rowBits = site(SystemBusKey).beatBits,
                nSets=%d,
                nWays=%d,
                fetchBytes=%d*4
              )
            )""" % (
            params[1],
            params[0],
            self.generate_fetch_width(vec, 1) << 1
        )

  def generate_dcache_and_mmu(self, vec, idx):
    params = self.get_mapping_params(vec, idx)
    decode_width = self.generate_decode_width(vec, 4)
    tlb_ways = [8, 8, 16, 32, 32]
    return """Some(
              DCacheParams(
                rowBits=site(SystemBusKey).beatBits,
                nSets=%d,
                nWays=%d,
                nMSHRs=%d,
                nTLBWays=%d
              )
            )""" % (
            params[1],
            params[0],
            params[2],
            tlb_ways[decode_width - 1]
        )

  def generate_core_cfg_impl(self, name, vec):
    codes = '''
class %s(n: Int = 1, overrideIdOffset: Option[Int] = None) extends Config(
  %s
  new Config((site, here, up) => {
    case TilesLocated(InSubsystem) => {
      val prev = up(TilesLocated(InSubsystem), site)
      val idOffset = overrideIdOffset.getOrElse(prev.size)
      (0 until n).map { i =>
        BoomTileAttachParams(
          tileParams = BoomTileParams(
            core = BoomCoreParams(
              fetchWidth = %d,
              numFetchBufferEntries = %d,
              ftq = FtqParameters(nEntries=%d),
              maxBrCount = %d,
              decodeWidth = %d,
              numRobEntries = %d,
              %s,
              issueParams = %s,
              %s,
              fpu = Some(
                freechips.rocketchip.tile.FPUParams(
                  sfmaLatency=4, dfmaLatency=4, divSqrt=true
                )
              ),
              enablePrefetching = true
            ),
            icache = %s,
            dcache = %s,
            hartId = i + idOffset
          ),
          crossingParams = RocketCrossingParams()
        )
      } ++ prev
    }
    case SystemBusKey => up(SystemBusKey, site).copy(beatBytes = %d)
    case XLen => 64
  })
)
''' % (
  name,
  self.generate_branch_predictor(vec, 0),
  self.generate_fetch_width(vec, 1),
  self.generate_fetch_buffer(vec, 2),
  self.generate_ftq(vec, 2),
  self.generate_max_br_count(vec, 3),
  self.generate_decode_width(vec, 4),
  self.generate_rob_entries(vec, 5),
  self.generate_phy_registers(vec, 6),
  self.generate_issue_parames(vec, 7),
  self.generate_lsu(vec, 8),
  self.generate_icache(vec, 9),
  self.generate_dcache_and_mmu(vec, 10),
  self.generate_fetch_width(vec, 1) << 1
)
    return codes

  def write_core_cfg_impl(self, codes):
    with open(self.macros["core-cfg"], 'a') as f:
      f.writelines(codes)

  def generate_soc_cfg_impl(self, soc_name, core_name):
    codes = '''
class %s extends Config(
  new boom.common.%s(1) ++
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
    # branch predictor
    embedding.append(vec[0])
    for idx in range(1, len(vec)):
      for v in self.get_mapping_params(vec, idx):
        embedding.append(v)
    return embedding

  def embedding_to_vec(self, embedding: List):
    """
        `embedding`: List
    """
    # branch predictor
    vec = []
    c, idx = 1, 1
    vec.append(embedding[0])
    while idx < len(embedding):
      prev_idx = idx
      num = len(self.components_mappings[self.components[c]]["description"])
      feature = embedding[idx : idx + num]
      for k, v in self.components_mappings[self.components[c]].items():
        if v == feature:
          vec.append(int(k))
          idx += num
          break
      if prev_idx == idx:
        break
      c += 1
    assert len(vec) == self.dims, assert_error(
          "invalid vec: {}".format(vec, idx)
        )
    return vec


class BOOMDesignSpace(DesignSpace, BOOMMacros):
  def __init__(self, root, descriptions, components_mappings, size):
    """
      descriptions: <class "collections.OrderedDict">
      Example:
      descriptions = {
        "1-wide SonicBOOM": {
          "fetchWidth": [4],
          "decodeWidth": [1],
          "enablePrefetching": [True],
          "ISU": [1, 2, 3],
          "IFU": [1, 2, 3],
          "ROB": [1, 2, 3, 4],
          "PRF": [1, 2, 3],
          "LSU": [1, 2, 3],
          "I-Cache/MMU": [1, 2, 3, 4],
          "D-Cache/MMU": [1, 2, 3, 4]
        }
      }

      components_mappings: <class "collections.OrderedDict">
      Example:
      components_mappings = {
        "ISU": {
          "description": ["IQT_MEM.dispatchWidth", "IQT_MEM.issueWidth"
            "IQT_MEM.numEntries", "IQT_INT.dispatchWidth",
            "IQT_INT.issueWidth", "IQT_INT.numEntries",
            "IQT_FP.dispatchWidth", "IQT_FP.issueWidth",
            "IQT_FP.numEntries"
          ],
          "1": [1, 1, 8, 1, 1, 8, 1, 1, 8],
          "2": [1, 1, 6, 1, 1, 6, 1, 1, 6]
        },
        "IFU": {
          "description": ["maxBrCount", "numFetchBufferEntries", "ftq.nEntries"]
          "1": [8, 8, 16],
          "2": [6, 6, 14]
        }
      }

      size: <int> the size of the entire design space
    """
    self.descriptions = descriptions
    self.components_mappings = components_mappings
    # construct look-up tables
    # self.designs: <list>
    # Example:
    #   ["1-wide SonicBOOM", "2-wide SonicBOOM", "3-wide SonicBOOM"]
    self.designs = list(self.descriptions.keys())
    # self.components: <list>
    # Example:
    #   ["fetchWidth", "decodeWidth", "ISU", "IFU"]
    self.components = list(self.descriptions[self.designs[0]].keys())
    # Example:
    #   length of ["branchPredictor", "fetchWidth", "numFetchBufferEntries", ...]
    self.embedding_dims = self.construct_embedding_dims()
    self.design_size = self.construct_design_size()
    # self.acc_design_size: <list> list of accumulated sizes of each design
    # Example:
    # self.design_size = [a, b, c, d, e] # accordingly
    #   self.acc_design_size = [a, a + b, a + b + c, a + b + c + d, a + b + c + d + e]
    self.acc_design_size = list(map(
        lambda x, idx: np.sum(x[:idx]),
        [self.design_size for i in range(len(self.design_size))],
        range(1, len(self.design_size) + 1)
      )
    )
    self.component_dims = self.construct_component_dims()
    self.type = [
      # "fetchWidth" "decodeWidth"
      [1, 1],
      [1, 2],
      [2, 3],
      [2, 4],
      [2, 5]
    ]
    DesignSpace.__init__(self, size, len(self.components))
    BOOMMacros.__init__(self, root)

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
    assert idx >= 0, \
        assert_error("invalid index: {}.".format(idx))
    assert idx < self.size, \
        assert_error("index exceeds the search space: {}.".format(idx))
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
    # adjust "fetchWidth" and "decodeWidth"
    vec[1] = self.type[design][0]
    vec[4] = self.type[design][1]
    return vec

  def vec_to_idx(self, vec):
    """
      vec: <list> the list of a microarchitecture encoding
    """

    idx = 0
    design = self.type.index([vec[1], vec[4]])
    # reset "fetchWidth" and "decodeWidth"
    for i in range(len(vec)):
      vec[i] = self.descriptions[self.designs[design]][self.components[i]].index(vec[i])
    vec[1], vec[4] = 0, 0
    for j, k in enumerate(vec):
        idx += int(np.prod(self.component_dims[design][:j])) * k
    if design >= 1:
      # NOTICE: add the offset
      idx += self.acc_design_size[design - 1]
    assert idx >= 0, \
        assert_error("invalid index: {}.".format(idx))
    assert idx < self.size, \
        assert_error("index exceeds the " \
            "search space: {}.".format(idx))
    idx += 1
    return idx

  def idx_to_embedding(self, idx):
    vec = self.idx_to_vec(idx)
    return self.vec_to_embedding(vec)

  def embedding_to_idx(self, microarchitecture_embedding):
    vec = self.embedding_to_vec(microarchitecture_embedding)
    return self.vec_to_idx(vec)

  def generate_core_cfg(self, batch):
    """
      generate core configurations
    """
    codes = []
    for idx in batch:
      codes.append(self.generate_core_cfg_impl(
          "WithN{}Booms".format(idx),
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
          "Boom{}Config".format(idx),
          "WithN{}Booms".format(idx)
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


def parse_boom_design_space(root, design_space_sheet, components_sheet):
  """
    design_space_sheet: <class "pandas.core.frame.DataFrame">
    components_sheet: <class "pandas.core.frame.DataFrame">
  """
  # parse design space
  descriptions = parse_design_space_sheet(design_space_sheet)

  # parse components
  components_mappings = parse_components_sheet(components_sheet)

  return BOOMDesignSpace(
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
  boom_design_space_sheet = load_excel(
    design_space_excel,
    sheet_name="BOOM Design Space"
  )
  components_sheet = load_excel(design_space_excel, sheet_name="Components")
  boom_design_space = parse_boom_design_space(
    configs["env"]["vlsi"]["chipyard-research-root"],
    boom_design_space_sheet,
    components_sheet
  )
  return boom_design_space
