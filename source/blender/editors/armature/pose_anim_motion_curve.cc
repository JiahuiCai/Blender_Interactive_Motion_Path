#include "MEM_guardedalloc.h"
extern "C" {

#include "DNA_action_types.h"
#include "DNA_anim_types.h"
#include "DNA_armature_types.h"
#include "DNA_curve_types.h"
#include "DNA_space_types.h"
#include "DNA_userdef_types.h"

#include "RNA_access.h"
#include "RNA_define.h"
#include "RNA_types.h"

#include "BLI_listbase.h"
#include "BLI_math.h"
#include "BLI_math_color.h"
#include "BLI_math_color_blend.h"

#include "BKE_action.h"
#include "BKE_callbacks.h"
#include "BKE_context.h"
#include "BKE_fcurve.h"
#include "BKE_object.h"
#include "BKE_scene.h"

#include "WM_api.h"
#include "WM_message.h"
#include "WM_toolsystem.h"
#include "WM_types.h"
#include "wm_event_types.h"
#include "wm_window.h"

#include "ED_anim_api.h"
#include "ED_gizmo_utils.h"
#include "ED_screen.h"
#include "ED_view3d.h"

#include "DEG_depsgraph.h"
#include "DEG_depsgraph_build.h"
#include "DEG_depsgraph_query.h"

#include "GPU_immediate.h"
#include "GPU_immediate_util.h"
#include "GPU_select.h"
#include "GPU_state.h"

#include "armature_intern.h"

#include "PIL_time.h"
}

#include <algorithm>
#include <array>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Dense"

// Notes:

const unsigned char RED[4] = {255, 0, 0, 255};
const unsigned char GREEN[4] = {0, 255, 0, 255};
const unsigned char BLUE[4] = {0, 0, 255, 255};
const unsigned char YELLOW[4] = {255, 255, 0, 255};
const unsigned char WHITE[4] = {255, 255, 255, 255};
const unsigned char PURPLE[4] = {255, 0, 255, 255};
const unsigned char CYAN[4] = {0, 255, 255, 255};

static const char *pose_gzgt_motion_curve_id = "POSE_GGT_motion_curve";

typedef struct _FrameData {
  bool is_keyframe;
  float ob_mat[4][4];
  float pose_head[3];
  float pose_tail[3];
  float gimbal[3][3];
  float local_mat[3][3];
  float quat[4];
  float pose_mat[4][4];
} FrameData;

typedef enum _FramePointType {
  HEAD,
  TAIL,
} FramePointType;

typedef struct _FramePoint {
  std::string ob_name;
  std::string pchan_name;
  float frame;
  FramePointType pt_type;
  float cached_pos[3];

  FrameData get_data();

  bool is_keyframe();

  void get_latest_pos(float r_pos[3]);

  bool operator==(struct _FramePoint const &pt) const;

  bool operator<(struct _FramePoint const &pt) const;

  friend std::ostream &operator<<(std::ostream &out, const struct _FramePoint &pt);
} FramePoint;

enum CurveType {
  LOC = 0,
  ROT_EUL = 1,
  CURVE_TYPE_MAX,
};

const char *CURVE_TYPE_NAME[CURVE_TYPE_MAX] = {"location", "rotation_euler"};

typedef struct _FCurveSegment {
  std::string ob_name;
  std::string pchan_name;
  FCurve *fcu;
  CurveType type;

  std::vector<int> keyframe_idx;

  float dc_dp(int i_keyIdx, int xy, float frame);

  bool operator==(struct _FCurveSegment const &seg) const;

  bool operator<(struct _FCurveSegment const &seg) const;

  friend std::ostream &operator<<(std::ostream &out, const struct _FCurveSegment &seg);

} FCurveSegment;

typedef struct MCTarget {
  FramePoint pt;
  float target[3];

  std::vector<FCurveSegment> all_segs;
  std::vector<FCurveSegment> pri_segs;
  std::vector<FCurveSegment> igr_segs;

  MCTarget(bContext *C, FramePoint pt, const float tar[3]);
} MCTarget;
typedef struct _MotionCurve {
  std::string ob_name;
  std::string pchan_name;
  bool is_highlight;
  int i_highlight_pt;
  std::vector<FramePoint> pt;
  float start;
  float end;
  void draw(int final_select_id);
  FramePoint get_selected_pt(float cfra);
} MotionCurve;

typedef struct _MotionCurveGlobals {
  bool is_init;
  bool is_updated;

  std::unordered_map<std::string, std::unordered_map<std::string, std::map<float, FrameData>>> fra;
  std::vector<MotionCurve> curves;
  std::set<FramePoint> pin;

  FramePoint select_pt;

} MotionCurveGlobals;

class MCSolver {
 public:
  int mode;
  std::vector<MCTarget> targets;
  std::vector<MCTarget> pins;
  std::vector<MCTarget> active_pins;
  std::vector<FCurveSegment> segs;

  void add_target(bContext *C, FramePoint pt, float target[3]);

  void add_pin(bContext *C, FramePoint pt, const float target[3]);

  void end_add();

  void update_target_interactive(bContext *C, wmGizmo *gz, const wmEvent *event);

  void solve(bContext *C);
};

typedef struct _MotionCurveItem {
  wmGizmo gz;
  MCSolver solver;
} MotionCurveItem;

namespace MC {
static MotionCurveGlobals G;
}
using namespace MC;

static void DEG_update(Depsgraph *depsgraph, Main *bmain);

static struct bToolRef *WM_toolsystem_ref_from_const_context(const struct bContext *C);

static Object *get_object_by_name(bContext *C, std::string name);

static bool my_quadprog(const Eigen::MatrixXd &Q,
                        const Eigen::VectorXd &c,
                        const std::vector<Eigen::Matrix3Xd> &Jc,
                        const std::vector<Eigen::Vector3d> &dP,
                        Eigen::VectorXd &x);

static void get_fcurve_segment_ex(std::vector<FCurveSegment> &segs,
                                  CurveType type,
                                  Object *ob,
                                  bPoseChannel *pchan,
                                  float frame);

static void get_sorted_fcurve_segment(bContext *C,
                                      std::vector<FCurveSegment> &segs,
                                      FramePoint pt);

static void get_sorted_primary_segments(bContext *C,
                                        std::vector<FCurveSegment> &segs,
                                        FramePoint pt);

bool FramePoint::operator==(struct _FramePoint const &pt) const
{
  return (ob_name == pt.ob_name) && (pchan_name == pt.pchan_name) && (frame == pt.frame) &&
         (pt_type == pt.pt_type);
}

bool FramePoint::operator<(struct _FramePoint const &pt) const
{
  if (ob_name != pt.ob_name) {
    return ob_name < pt.ob_name;
  }

  if (pchan_name != pt.pchan_name) {
    return pchan_name < pt.pchan_name;
  }

  if (frame != pt.frame) {
    return frame < pt.frame;
  }

  if (pt_type != pt.pt_type) {
    return pt_type < pt.pt_type;
  }

  return false;
}

std::ostream &operator<<(std::ostream &out, const struct _FramePoint &pt)
{
  out << pt.ob_name << " " << pt.pchan_name << " " << (pt.pt_type == HEAD ? "HEAD" : "TAIL") << " "
      << pt.frame << std::endl;

  return out;
}

float FCurveSegment::dc_dp(int i_keyIdx, int xy, float frame)
{
  BLI_assert(i_keyIdx < keyframe_idx.size());

  int i_k = keyframe_idx[i_keyIdx];

  float dCdT = 0;
  float dT = 0.000001;

  float C1 = 0;
  float C2 = 0;

  int i_tan = -1;
  if (keyframe_idx.size() == 1) {
    BLI_assert(xy == 1);
    i_tan = 1;
  }
  else if (keyframe_idx.size() == 2) {
    i_tan = i_keyIdx == 0 ? 2 : 0;
  }
  else {
    BLI_assert(false);
  }

  fcu->bezt[i_k].vec[i_tan][xy] += dT;
  C1 = evaluate_fcurve(fcu, frame);
  fcu->bezt[i_k].vec[i_tan][xy] -= dT;

  fcu->bezt[i_k].vec[i_tan][xy] -= dT;
  C2 = evaluate_fcurve(fcu, frame);
  fcu->bezt[i_k].vec[i_tan][xy] += dT;

  dCdT = (C1 - C2) / (2 * dT);

  return dCdT;
}

bool FCurveSegment::operator==(struct _FCurveSegment const &seg) const
{
  return (ob_name == seg.ob_name) && (pchan_name == seg.pchan_name) && (fcu == seg.fcu) &&
         (fcu->bezt[keyframe_idx[0]].vec[1][0] == seg.fcu->bezt[seg.keyframe_idx[0]].vec[1][0]) &&
         (keyframe_idx.size() == keyframe_idx.size());
}

bool FCurveSegment::operator<(struct _FCurveSegment const &seg) const
{
  if (ob_name > seg.ob_name) {
    return false;
  }
  else if (ob_name < seg.ob_name) {
    return true;
  }

  if (pchan_name > seg.pchan_name) {
    return false;
  }
  else if (pchan_name < seg.pchan_name) {
    return true;
  }

  if (fcu > seg.fcu) {
    return false;
  }
  else if (fcu < seg.fcu) {
    return true;
  }

  if (fcu->bezt[keyframe_idx[0]].vec[1][0] > seg.fcu->bezt[seg.keyframe_idx[0]].vec[1][0]) {
    return false;
  }
  else if (fcu->bezt[keyframe_idx[0]].vec[1][0] < seg.fcu->bezt[seg.keyframe_idx[0]].vec[1][0]) {
    return true;
  }

  if (keyframe_idx.size() > seg.keyframe_idx.size()) {
    return false;
  }
  else if (keyframe_idx.size() < seg.keyframe_idx.size()) {
    return true;
  }

  return false;
}

std::ostream &operator<<(std::ostream &out, const struct _FCurveSegment &seg)
{
  out << seg.fcu->rna_path << " " << seg.fcu->array_index << std::endl;
  for (auto i : seg.keyframe_idx) {
    out << seg.fcu->bezt[i].vec[1][0] << " ";
  }
  out << std::endl;

  return out;
}

MCTarget::MCTarget(bContext *C, FramePoint pt, const float tar[3])
{
  this->pt = pt;
  copy_v3_v3(this->target, tar);

  std::cout << "********Target Created:" << std::endl;
  std::cout << pt;
  print_v3("tar:", tar);

  get_sorted_fcurve_segment(C, this->all_segs, pt);
  get_sorted_primary_segments(C, this->pri_segs, pt);

  std::set_difference(all_segs.begin(),
                      all_segs.end(),
                      pri_segs.begin(),
                      pri_segs.end(),
                      std::back_inserter(igr_segs));

  std::cout << "pri segs:" << std::endl;
  for (auto seg : pri_segs) {
    std::cout << seg;
  }
  std::cout << "_________Target Created:" << std::endl;
}

FramePoint MotionCurve::get_selected_pt(float cfra)
{
  FramePoint hl_pt = pt[i_highlight_pt];

  if (cfra <= hl_pt.frame) {
    return hl_pt;
  }
  else {
    // if it is not the last point
    FramePoint sel_pt = hl_pt;

    if (i_highlight_pt < pt.size() - 1) {
      FramePoint nx_pt = pt[i_highlight_pt + 1];
      if (cfra >= nx_pt.frame) {
        sel_pt.frame = nx_pt.frame - 1;
      }
      else {
        sel_pt.frame = cfra;
      }
    }
    else {
      sel_pt.frame = cfra;
    }

    return sel_pt;
  }
}

FrameData FramePoint::get_data()
{
  return G.fra[ob_name][pchan_name][frame];
}

bool FramePoint::is_keyframe()
{
  return G.fra[ob_name][pchan_name][frame].is_keyframe;
}

void FramePoint::get_latest_pos(float r_pos[3])
{
  switch (pt_type) {
    case HEAD:
      copy_v3_v3(r_pos, get_data().pose_head);
      break;
    case TAIL:
      copy_v3_v3(r_pos, get_data().pose_tail);
      break;
    default:
      BLI_assert(false);
      break;
  }
}

void MotionCurve::draw(int final_select_id)
{
  GPU_select_load_id(final_select_id);

  if (pt.size() > 1) {

    GPU_blend(true);
    GPU_blend_set_func_separate(
        GPU_SRC_ALPHA, GPU_ONE_MINUS_SRC_ALPHA, GPU_ONE, GPU_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);
    GPU_line_smooth(true);

    GPUVertFormat *format = immVertexFormat();
    uint pos = GPU_vertformat_attr_add(format, "pos", GPU_COMP_F32, 3, GPU_FETCH_FLOAT);
    uint col = GPU_vertformat_attr_add(
        format, "color", GPU_COMP_U8, 4, GPU_FETCH_INT_TO_FLOAT_UNIT);
    immBindBuiltinProgram(GPU_SHADER_3D_POLYLINE_SMOOTH_COLOR);

    float viewport[4];
    GPU_viewport_size_get_f(viewport);
    immUniform2fv("viewportSize", &viewport[2]);

    const unsigned char *col_start = YELLOW;
    const unsigned char *col_end = PURPLE;
    float line_width = 2.0;
    if (is_highlight) {
      col_start = GREEN;
      col_end = BLUE;
      line_width = 6.0;
    }

    if (final_select_id >= 0) {
      // thin line for selection, thick line for visualization
      line_width = 1.0;
    }

    immUniform1f("lineWidth", line_width * U.pixelsize);
    immBegin(GPU_PRIM_LINE_STRIP, pt.size());

    // draw the curve
    for (int i = 0; i < pt.size(); i++) {
      float progression = (float)i / (pt.size() - 1);

      unsigned char dst[4];
      blend_color_interpolate_byte(dst, col_start, col_end, progression);

      FramePoint p = pt[i];
      float pt_pos[3];
      p.get_latest_pos(pt_pos);

      immAttr4ubv(col, dst);
      immVertex3fv(pos, pt_pos);
    }

    immEnd();
    immUnbindProgram();

    GPU_blend(false);
    glDepthMask(GL_TRUE);
    GPU_line_smooth(false);
    /* Reset default. */
    GPU_blend_set_func_separate(
        GPU_SRC_ALPHA, GPU_ONE_MINUS_SRC_ALPHA, GPU_ONE, GPU_ONE_MINUS_SRC_ALPHA);
  }

  // don use points for selection
  {
    GPU_program_point_size(true);
    GPUVertFormat *format = immVertexFormat();
    uint pos_attr = GPU_vertformat_attr_add(format, "pos", GPU_COMP_F32, 3, GPU_FETCH_FLOAT);
    uint size_attr = GPU_vertformat_attr_add(format, "size", GPU_COMP_F32, 1, GPU_FETCH_FLOAT);
    uint color_attr = GPU_vertformat_attr_add(
        format, "color", GPU_COMP_U8, 4, GPU_FETCH_INT_TO_FLOAT_UNIT);

    immBindBuiltinProgram(GPU_SHADER_3D_POINT_VARYING_SIZE_VARYING_COLOR);
    immBegin(GPU_PRIM_POINTS, pt.size());

    const unsigned char *col_start = YELLOW;
    const unsigned char *col_end = PURPLE;
    if (is_highlight) {
      col_start = GREEN;
      col_end = BLUE;
    }

    // draw the curve
    for (int i = 0; i < pt.size(); i++) {
      float progression = ((float)i) / (float)(pt.size() - 1);

      const unsigned char *final_col;
      unsigned char dst[4];
      blend_color_interpolate_byte(dst, col_start, col_end, progression);
      final_col = dst;

      FramePoint p = pt[i];
      float pos[3];
      p.get_latest_pos(pos);

      float size = 4.0;

      float frame = p.frame;

      if (p.is_keyframe()) {
        size = 8.0;
      }

      if (i == i_highlight_pt && is_highlight) {
        size = 12.0;
      }

      auto iter = std::find(G.pin.begin(), G.pin.end(), p);
      if (iter != G.pin.end()) {
        final_col = RED;
        size = 20.0;
      }

      if (final_select_id != -1) {
        size = 3.0;
      }

      immAttr4ubv(color_attr, final_col);
      immAttr1f(size_attr, size);
      immVertex3fv(pos_attr, pos);
    }

    immEnd();
    immUnbindProgram();
    GPU_program_point_size(false);
  }
}

void MCSolver::add_target(bContext *C, FramePoint pt, float target[3])
{
  MCTarget mc_tar(C, pt, target);
  this->targets.push_back(mc_tar);
};

void MCSolver::add_pin(bContext *C, FramePoint pt, const float target[3])
{
  MCTarget mc_pin(C, pt, target);
  this->pins.push_back(mc_pin);
};

void MCSolver::end_add()
{

  std::vector<FCurveSegment> all_segs;
  std::vector<FCurveSegment> ignored_segs;

  for (auto &tar : targets) {
    std::vector<FCurveSegment> new_all_segs;
    std::set_union(all_segs.begin(),
                   all_segs.end(),
                   tar.all_segs.begin(),
                   tar.all_segs.end(),
                   std::back_inserter(new_all_segs));
    all_segs = new_all_segs;

    std::vector<FCurveSegment> new_ignored_segs;
    std::set_union(ignored_segs.begin(),
                   ignored_segs.end(),
                   tar.igr_segs.begin(),
                   tar.igr_segs.end(),
                   std::back_inserter(new_ignored_segs));
    ignored_segs = new_ignored_segs;
  }

  std::vector<bool> pin_disabled;

  // detect pins in the parents
  for (auto &pin : pins) {
    bool is_disabled = false;
    if (std::includes(
            all_segs.begin(), all_segs.end(), pin.all_segs.begin(), pin.all_segs.end())) {
      if (all_segs.size() > pin.all_segs.size()) {

        std::vector<FCurveSegment> new_ignored_segs;
        std::set_union(ignored_segs.begin(),
                       ignored_segs.end(),
                       pin.all_segs.begin(),
                       pin.all_segs.end(),
                       std::back_inserter(new_ignored_segs));
        ignored_segs = new_ignored_segs;
        is_disabled = true;
      }
    }
    pin_disabled.push_back(is_disabled);
  }

  // ignored_segs is now completed

  // use ignored_segs to reduce primary_segs
  std::vector<FCurveSegment> total_reduced_primary_segs;
  for (auto &tar : targets) {
    std::vector<FCurveSegment> reduced_primary_segs;
    std::set_difference(tar.pri_segs.begin(),
                        tar.pri_segs.end(),
                        ignored_segs.begin(),
                        ignored_segs.end(),
                        std::back_inserter(reduced_primary_segs));
    tar.pri_segs = reduced_primary_segs;

    std::vector<FCurveSegment> total_reduced_primary_segs_accu;
    std::set_union(total_reduced_primary_segs.begin(),
                   total_reduced_primary_segs.end(),
                   tar.pri_segs.begin(),
                   tar.pri_segs.end(),
                   std::back_inserter(total_reduced_primary_segs_accu));
    total_reduced_primary_segs = total_reduced_primary_segs_accu;
  }

  segs = total_reduced_primary_segs;

  // reduce segs for pin in the child or sibings, potentially disable it
  for (int i = 0; i < pins.size(); i++) {
    bool is_disabled = pin_disabled[i];

    if (!is_disabled) {
      is_disabled = true;
      std::vector<FCurveSegment> reduced_pin_segs;
      std::set_difference(pins[i].all_segs.begin(),
                          pins[i].all_segs.end(),
                          ignored_segs.begin(),
                          ignored_segs.end(),
                          std::back_inserter(reduced_pin_segs));

      std::vector<FCurveSegment> shared_segs;

      std::set_intersection(all_segs.begin(),
                            all_segs.end(),
                            reduced_pin_segs.begin(),
                            reduced_pin_segs.end(),
                            std::back_inserter(shared_segs));

      if (shared_segs.size() > 0) {
        is_disabled = false;
        // rigid means that we only use joint rotation for solver,
        // a bone cannot change its location during interaction
        std::vector<FCurveSegment> rigid_pin_segs;
        for (auto &seg : reduced_pin_segs) {
          if (seg.type == ROT_EUL) {
            rigid_pin_segs.push_back(seg);
          }
          else if (seg.type == LOC) {
            if (targets[0].pt.pt_type == HEAD) {
              auto iter = std::find(
                  total_reduced_primary_segs.begin(), total_reduced_primary_segs.end(), seg);
              if (iter != total_reduced_primary_segs.end()) {
                // only include location curve that the user is interacting
                rigid_pin_segs.push_back(seg);
              }
            }
          }
        }

        pins[i].pri_segs = rigid_pin_segs;
      }
    }
    pin_disabled[i] = is_disabled;
  }

  for (int i = 0; i < pins.size(); i++) {
    bool is_disabled = pin_disabled[i];
    if (!is_disabled) {
      active_pins.push_back(pins[i]);
    }
  }

  for (auto &pin : active_pins) {
    std::vector<FCurveSegment> augmented_segs;
    std::set_union(segs.begin(),
                   segs.end(),
                   pin.pri_segs.begin(),
                   pin.pri_segs.end(),
                   std::back_inserter(augmented_segs));

    segs = augmented_segs;
  }
}

void MCSolver::update_target_interactive(bContext *C, wmGizmo *gz, const wmEvent *event)
{
  static float last_mvalf[2] = {0, 0};
  float ref_pos[3];
  targets[0].pt.get_latest_pos(ref_pos);

  View3D *v3d = CTX_wm_view3d(C);
  ARegion *ar = CTX_wm_region(C);
  float zfac = ED_view3d_calc_zfac((RegionView3D *)ar->regiondata, ref_pos, NULL);
  float delta[3] = {0, 0, 0};

  float mval_fl[2];

  float mval_del[2] = {event->x - event->prevx, event->y - event->prevy};

  ED_view3d_win_to_delta(ar, mval_del, delta, zfac);

  add_v3_v3v3(targets[0].target, ref_pos, delta);
}

void MCSolver::solve(bContext *C)
{
  if (segs.size() > 0) {
    int num_param_per_seg = segs[0].keyframe_idx.size() == 1 ? 1 : 4;
    int num_param_total = segs.size() * num_param_per_seg;

    std::vector<Eigen::Matrix3Xd> J(targets.size(), Eigen::Matrix3Xd::Zero(3, num_param_total));

    // I am exploiting the fact that all vector of segs are sorted

    for (int i_t = 0; i_t < targets.size(); i_t++) {
      MCTarget &tar = targets[i_t];
      Eigen::Matrix3Xd &j = J[i_t];  // 3 * num_param
      int i_ts = 0;
      for (int i_s = 0; i_s < segs.size(); i_s++) {
        if (i_ts < tar.pri_segs.size()) {
          if (tar.pri_segs[i_ts] == segs[i_s]) {

            Eigen::Vector3d djdc;

            float frame = tar.pt.frame;
            std::string ob_name = segs[i_s].ob_name;
            std::string pchan_name = segs[i_s].pchan_name;
            float tar_pt_pos[3];
            tar.pt.get_latest_pos(tar_pt_pos);

            FrameData data = G.fra[ob_name][pchan_name][frame];

            int axis = segs[i_s].fcu->array_index;

            switch (segs[i_s].type) {
              case ROT_EUL: {
                // tail control only applies to chain rotation
                if (tar.pt.pt_type == TAIL) {
                  float rot_arm[4] = {};
                  sub_v4_v4v4(rot_arm, tar_pt_pos, data.pose_head);

                  float rot_dir[4] = {};
                  cross_v3_v3v3(rot_dir, data.gimbal[axis], rot_arm);

                  djdc = Eigen::Map<Eigen::Vector3f>(rot_dir).cast<double>();
                }
              } break;
              case LOC: {
                // head control only applies to the selected bone
                if (tar.pt.pt_type == HEAD) {
                  if (segs[i_s].pchan_name == tar.pt.pchan_name) {
                    djdc = Eigen::Map<Eigen::Vector3f>(data.local_mat[axis]).cast<double>();
                  }
                }
              } break;
              default:
                break;
            }

            if (num_param_per_seg == 1) {
              j.col(i_s) = djdc;
            }
            else if (num_param_per_seg == 4) {
              j.col(i_s * num_param_per_seg + 0) = djdc * segs[i_s].dc_dp(0, 0, frame);
              j.col(i_s * num_param_per_seg + 1) = djdc * segs[i_s].dc_dp(0, 1, frame);
              j.col(i_s * num_param_per_seg + 2) = djdc * segs[i_s].dc_dp(1, 0, frame);
              j.col(i_s * num_param_per_seg + 3) = djdc * segs[i_s].dc_dp(1, 1, frame);
            }
            else {
              BLI_assert(false);
            }

            i_ts++;
          }
        }
      }
    }

    double w1 = 100;
    double wd = 1;
    // std::cout << "J[0]" << std::endl;
    // std::cout << J[0] << std::endl;
    // this multipication of 2 needs to go outside
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(num_param_total, num_param_total);

    for (auto &j : J) {
      Q += j.transpose() * j * w1;
    }

    Q += Eigen::MatrixXd::Identity(num_param_total, num_param_total) * wd;
    Q *= 2;

    Eigen::VectorXd c = Eigen::VectorXd::Zero(num_param_total);
    for (int i_t = 0; i_t < targets.size(); i_t++) {
      Eigen::Vector3d ds;
      float _ds[3];
      float tar_pt_pos[3];
      targets[i_t].pt.get_latest_pos(tar_pt_pos);
      sub_v3_v3v3(_ds, targets[i_t].target, tar_pt_pos);

      if (this->mode == 1) {
        if (len_v3(_ds) > 0.01) {
          normalize_v3_length(_ds, 0.01);
        }
      }

      ds << _ds[0], _ds[1], _ds[2];

      c += ((-2) * w1 * (ds.transpose() * J[i_t])).transpose();
    }

    Eigen::VectorXd x = Eigen::VectorXd::Zero(num_param_total);

    std::vector<Eigen::Matrix3Xd> v_Jc;

    std::vector<Eigen::Vector3d> v_dP;

    for (int i = 0; i < active_pins.size(); i++) {
      MCTarget pin = active_pins[i];
      FramePoint pin_pt = pin.pt;
      FrameData cur_pin_data = pin_pt.get_data();

      float pin_frame = pin_pt.frame;
      FramePointType pin_type = pin_pt.pt_type;

      const float *cur_pin_pos;
      switch (pin_type) {
        case HEAD:
          cur_pin_pos = cur_pin_data.pose_head;
          break;
        case TAIL:
          cur_pin_pos = cur_pin_data.pose_tail;
          break;
      }

      float dP_[3];
      sub_v3_v3v3(dP_, pin.target, cur_pin_pos);

      Eigen::Vector3d dP;
      dP << dP_[0], dP_[1], dP_[2];
      v_dP.push_back(dP);

      Eigen::Matrix3Xd Jc = Eigen::MatrixXd::Zero(3, num_param_total);

      std::vector<FCurveSegment> &pin_segs = pin.pri_segs;
      int i_ps = 0;
      int i_s = 0;
      while (i_ps < pin_segs.size() && i_s < segs.size()) {
        FCurveSegment seg = segs[i_s];
        if (pin_segs[i_ps] == seg) {
          FrameData data = G.fra[seg.ob_name][seg.pchan_name][pin_frame];

          FCurve *fcu = seg.fcu;
          int axis = fcu->array_index;

          Eigen::Vector3d dJdv;
          dJdv << 0.0, 0.0, 0.0;

          switch (seg.type) {
            case ROT_EUL: {
              float rot_arm[4] = {};
              sub_v4_v4v4(rot_arm, pin.target, data.pose_head);

              float rot_dir[4] = {};
              cross_v3_v3v3(rot_dir, data.gimbal[axis], rot_arm);

              dJdv = Eigen::Map<Eigen::Vector3f>(rot_dir).cast<double>();
            } break;
            case LOC: {
              dJdv = Eigen::Map<Eigen::Vector3f>(data.local_mat[axis]).cast<double>();
            } break;
            default:
              break;
          }

          if (seg.keyframe_idx.size() == 1) {
            Jc.col(i_s) = dJdv;
          }
          else if (seg.keyframe_idx.size() == 2) {
            Jc.col(i_s * num_param_per_seg + 0) = dJdv * segs[i_s].dc_dp(0, 0, pin_frame);
            Jc.col(i_s * num_param_per_seg + 1) = dJdv * segs[i_s].dc_dp(0, 1, pin_frame);
            Jc.col(i_s * num_param_per_seg + 2) = dJdv * segs[i_s].dc_dp(1, 0, pin_frame);
            Jc.col(i_s * num_param_per_seg + 3) = dJdv * segs[i_s].dc_dp(1, 1, pin_frame);
          }

          i_ps++;
        }
        i_s++;
      }

      v_Jc.push_back(Jc);
    }

    my_quadprog(Q, c, v_Jc, v_dP, x);

    if (num_param_per_seg == 1) {
      for (int i_s = 0; i_s < segs.size(); i_s++) {
        FCurveSegment &seg = segs[i_s];
        seg.fcu->bezt[seg.keyframe_idx[0]].h1 = HD_FREE;
        seg.fcu->bezt[seg.keyframe_idx[0]].h2 = HD_FREE;
        seg.fcu->bezt[seg.keyframe_idx[0]].vec[1][1] += x(i_s);
      }
    }
    else {
      for (int i_s = 0; i_s < segs.size(); i_s++) {
        FCurveSegment &seg = segs[i_s];
        seg.fcu->bezt[seg.keyframe_idx[0]].h1 = HD_FREE;
        seg.fcu->bezt[seg.keyframe_idx[0]].h2 = HD_FREE;

        seg.fcu->bezt[seg.keyframe_idx[1]].h1 = HD_FREE;
        seg.fcu->bezt[seg.keyframe_idx[1]].h2 = HD_FREE;

        seg.fcu->bezt[seg.keyframe_idx[0]].vec[2][0] += x(i_s * num_param_per_seg + 0);
        seg.fcu->bezt[seg.keyframe_idx[0]].vec[2][1] += x(i_s * num_param_per_seg + 1);
        seg.fcu->bezt[seg.keyframe_idx[1]].vec[0][0] += x(i_s * num_param_per_seg + 2);
        seg.fcu->bezt[seg.keyframe_idx[1]].vec[0][1] += x(i_s * num_param_per_seg + 3);
      }
    }

    for (int i_s = 0; i_s < segs.size(); i_s++) {
      FCurveSegment &seg = segs[i_s];
      PointerRNA id_ptr, ptr;
      PropertyRNA *prop;

      Object *seg_ob = get_object_by_name(C, seg.ob_name);
      BLI_assert(seg_ob != NULL);

      RNA_id_pointer_create((ID *)seg_ob, &id_ptr);

      if (RNA_path_resolve_property(&id_ptr, seg.fcu->rna_path, &ptr, &prop)) {
        Scene *scene = CTX_data_scene(C);
        Main *bmain = CTX_data_main(C);
        RNA_property_update_main(bmain, scene, &ptr, prop);
      }
    }

    Object *ob = get_object_by_name(C, targets[0].pt.ob_name);
    DEG_id_tag_update((ID *)ob, ID_RECALC_ANIMATION);
    if (ob->adt->action != NULL) {
      DEG_id_tag_update(&ob->adt->action->id, ID_RECALC_ANIMATION);
    }

    ED_region_tag_redraw(CTX_wm_region(C));
  }
}

static void WIDGETGROUP_motion_curve_setup(const bContext *C, wmGizmoGroup *gzgroup)
{
  if (G.is_init == false) {
    G.is_init = true;
  }

  wmGizmo *gz = WM_gizmo_new("POSE_GT_motion_curve_item", gzgroup, NULL);
}

static void WIDGETGROUP_motion_curve_refresh(const struct bContext *C,
                                             struct wmGizmoGroup *gzgroup)
{
  if (G.is_updated) {
    return;
  }

  Scene *scene = CTX_data_scene(C);
  Main *bmain = CTX_data_main(C);
  ViewLayer *view_layer = CTX_data_view_layer(C);

  // Collect all objects that could be updated potentially
  std::set<Object *> arm_objs;
  {

    ListBase selected_pose_bones;
    if (CTX_data_selected_pose_bones(C, &selected_pose_bones) == 0) {
      return;
    }

    LISTBASE_FOREACH (CollectionPointerLink *, link, &selected_pose_bones) {
      bPoseChannel *pchan = (bPoseChannel *)link->ptr.data;
      Object *ob = (Object *)link->ptr.owner_id;
      arm_objs.insert(ob);
    }
    BLI_freelistN(&selected_pose_bones);
  }

  // Go through each obj and update its data at each frame
  G.fra.clear();

  float cfra = CFRA;
  // double time_start = PIL_check_seconds_timer();
  bToolRef *tref = WM_toolsystem_ref_from_const_context(C);

  PointerRNA gzg_ptr;
  WM_toolsystem_ref_properties_ensure_from_gizmo_group(tref, gzgroup->type, &gzg_ptr);
  const int range = RNA_int_get(&gzg_ptr, "range");

  for (Object *ob : arm_objs) {  // Get start and end frame of the current interpolation;
    float start = -1;
    int start_keyframe_idx = -1;
    float end = -2;
    int end_keyframe_idx = -1;

    bool is_editable = false;
    std::vector<float> keyframes;
    {
      bAction *act = ob->adt->action;

      if (act == NULL) {
        continue;
      }

      FCurve *fcu = NULL;
      for (fcu = (FCurve *)act->curves.first; fcu; fcu = fcu->next) {
        if (fcu->totvert) {
          break;
        }
      }

      if (fcu == NULL) {
        continue;
      }

      if (fcu->totvert > 0) {
        for (int x = 0; x < fcu->totvert; x++) {
          float key_time = fcu->bezt[x].vec[1][0];
          keyframes.push_back(key_time);
        }
      }

      if (keyframes.size() > 0 && cfra >= keyframes.front() && cfra <= keyframes.back()) {
        is_editable = true;

        int i = 0;
        while (i < keyframes.size() && keyframes[i] < cfra) {
          i++;
        }

        float key_time = keyframes[i];

        if (i - range >= 0) {
          // keyframe on the left side
          start = keyframes[i - range];
          start_keyframe_idx = i - range;
        }
        else {
          start = keyframes[0];
          start_keyframe_idx = 0;
        }

        // keyframe on the right side
        if (key_time == cfra) {
          if (i + range < keyframes.size()) {
            end = keyframes[i + range];
            end_keyframe_idx = i + range;
          }
          else {
            end = keyframes[keyframes.size() - 1];
            end_keyframe_idx = keyframes.size() - 1;
          }
        }
        else if (key_time > cfra) {
          if (i + range - 1 < keyframes.size()) {

            end = keyframes[i + range - 1];
            end_keyframe_idx = i + range - 1;
          }
          else {
            end = keyframes[keyframes.size() - 1];
            end_keyframe_idx = keyframes.size() - 1;
          }
        }
      }
    }

    if (is_editable) {
      /* Allocate dependency graph. */
      Depsgraph *depsgraph = DEG_graph_new(bmain, scene, view_layer, DAG_EVAL_VIEWPORT);
      ID **ids = (ID **)MEM_malloc_arrayN(sizeof(ID *), 1, "object id");
      ids[0] = &(ob->id);

      /* Build graph from all requested IDs. */
      DEG_graph_build_from_ids(depsgraph, bmain, scene, view_layer, ids, 1);
      MEM_freeN(ids);

      int i_keyframes = start_keyframe_idx;
      for (CFRA = start; CFRA <= end; CFRA++) {
        bool is_keyframe = false;

        if (CFRA == keyframes[i_keyframes]) {
          is_keyframe = true;
          i_keyframes++;
        }

        DEG_update(depsgraph, bmain);
        for (bPoseChannel *pchan = (bPoseChannel *)ob->pose->chanbase.first; pchan;
             pchan = pchan->next) {
          Object *ob_eval = DEG_get_evaluated_object(depsgraph, ob);
          bPoseChannel *pchan_eval = BKE_pose_channel_find_name(ob_eval->pose, pchan->name);

          float gim_mat[3][3];
          float local_mat[3][3];
          {
            eulO_to_gimbal_axis(gim_mat, pchan_eval->eul, pchan_eval->rotmode);
            mul_m3_m3m3(gim_mat, pchan_eval->bone->bone_mat, gim_mat);
            copy_m3_m3(local_mat, pchan_eval->bone->bone_mat);
            if (pchan->parent) {
              float parent_mat[3][3];

              bPoseChannel *parent_pchan_eval = BKE_pose_channel_find_name(ob_eval->pose,
                                                                           pchan->parent->name);
              copy_m3_m4(parent_mat, parent_pchan_eval->pose_mat);
              mul_m3_m3m3(gim_mat, parent_mat, gim_mat);
              mul_m3_m3m3(local_mat, parent_mat, local_mat);
            }
            normalize_m3(local_mat);
            normalize_m3(gim_mat);
          }

          FrameData fra_data;
          fra_data.is_keyframe = is_keyframe;
          copy_m4_m4(fra_data.ob_mat, ob_eval->obmat);
          copy_v3_v3(fra_data.pose_tail, pchan_eval->pose_tail);
          copy_v3_v3(fra_data.pose_head, pchan_eval->pose_head);
          copy_m3_m3(fra_data.gimbal, gim_mat);
          copy_m3_m3(fra_data.local_mat, local_mat);
          copy_v4_v4(fra_data.quat, pchan_eval->quat);
          copy_m4_m4(fra_data.pose_mat, pchan_eval->pose_mat);

          G.fra[ob->id.name][pchan->name][CFRA] = fra_data;
        }
      }

      DEG_graph_free(depsgraph);
    }
  }
  CFRA = cfra;

  // Update the visualization of the motin path
  G.curves.clear();
  {
    ListBase selected_pose_bones;
    CTX_data_selected_pose_bones(C, &selected_pose_bones);

    LISTBASE_FOREACH (CollectionPointerLink *, link, &selected_pose_bones) {
      Object *ob = (Object *)link->ptr.owner_id;
      bPoseChannel *pchan = (bPoseChannel *)link->ptr.data;

      std::map<float, FrameData> &h_fra = G.fra[ob->id.name][pchan->name];

      if (h_fra.size() == 0) {
        continue;
      }

      int num_curves = 1;
      if ((pchan->bone->flag & BONE_CONNECTED) != BONE_CONNECTED) {
        num_curves = 2;
      }

      PointerRNA ptr;
      RNA_pointer_create((ID *)ob, &RNA_PoseBone, pchan, &ptr);

      bool is_show_head = RNA_boolean_get(&ptr, "show_head_curve");
      bool is_show_tail = RNA_boolean_get(&ptr, "show_tail_curve");

      for (int i_curve = 0; i_curve < num_curves; i_curve++) {
        if (!is_show_tail && i_curve == 0) {
          continue;
        }
        if (!is_show_head && i_curve == 1) {
          continue;
        }

        std::vector<FramePoint> pts;
        for (auto &pair : h_fra) {
          float frame = pair.first;
          FramePoint pt;
          pt.ob_name = std::string(ob->id.name);
          pt.pchan_name = std::string(pchan->name);
          pt.frame = frame;

          switch (i_curve) {
            case 0:
              pt.pt_type = TAIL;
              break;
            case 1:
              pt.pt_type = HEAD;
              break;
          }
          pts.push_back(pt);
        }

        int first = 0;
        int last = pts.size() - 1;

        G.curves.push_back(MotionCurve());
        G.curves.back().ob_name = std::string(ob->id.name);
        G.curves.back().pchan_name = std::string(pchan->name);
        if (pts.size() > 0) {
          G.curves.back().start = pts.front().frame;
          G.curves.back().end = pts.back().frame;
        }
        else {
          G.curves.back().start = -1;
          G.curves.back().end = -1;
        }

        for (int i = first; i <= last; i++) {
          if (i == 0) {
            G.curves.back().pt.push_back(pts[i]);
          }
          else {
            float cur_pos[3];
            float las_pos[3];
            pts[i].get_latest_pos(cur_pos);
            pts[i - 1].get_latest_pos(las_pos);

            bool is_overlapped = equals_v3v3(cur_pos, las_pos);

            if (!is_overlapped) {
              G.curves.back().pt.push_back(pts[i]);
            }
          }
        }
      }
    }

    BLI_freelistN(&selected_pose_bones);
  }

  G.is_updated = true;
}

static void motion_curve_property_update()
{
  G.is_updated = false;
}

static void WIDGETGROUP_motion_curve_message_subscribe(const bContext *C,
                                                       wmGizmoGroup *gzgroup,
                                                       wmMsgBus *mbus)
{
  ARegion *region = CTX_wm_region(C);

  {
    wmMsgSubscribeValue msg_sub_value_gz_tag_refresh = {0};
    msg_sub_value_gz_tag_refresh.owner = region;
    msg_sub_value_gz_tag_refresh.user_data = gzgroup->parent_gzmap,
    msg_sub_value_gz_tag_refresh.notify = WM_gizmo_do_msg_notify_tag_refresh;

    // double time_start = PIL_check_seconds_timer();
    bToolRef *tref = WM_toolsystem_ref_from_const_context(C);
    PointerRNA gzg_ptr;
    WM_toolsystem_ref_properties_ensure_from_gizmo_group(tref, gzgroup->type, &gzg_ptr);

    WM_msg_subscribe_rna(mbus, &gzg_ptr, NULL, &msg_sub_value_gz_tag_refresh, __func__);
  }

  {
    wmMsgSubscribeValue msg_sub_value_gz_tag_refresh = {0};
    msg_sub_value_gz_tag_refresh.owner = region;
    msg_sub_value_gz_tag_refresh.user_data = gzgroup->parent_gzmap,
    msg_sub_value_gz_tag_refresh.notify = WM_gizmo_do_msg_notify_tag_refresh;

    // double time_start = PIL_check_seconds_timer();
    StructRNA *srna = RNA_struct_find("PoseBone");

    PointerRNA ptr = {0};
    ptr.type = srna;
    wmMsgParams_RNA param = {0};

    param.ptr = ptr;
    param.prop = NULL;

    WM_msg_subscribe_rna_params(mbus, &param, &msg_sub_value_gz_tag_refresh, __func__);
  }
}

static void WIDGETGROUP_motion_curve_draw_prepare(const bContext *C, wmGizmoGroup *gzgroup)
{
}

void POSE_GGT_motion_curve(wmGizmoGroupType *gzgt)
{
  gzgt->name = "Motion Curve Widgets";
  gzgt->idname = pose_gzgt_motion_curve_id;

  int flag = gzgt->flag | WM_GIZMOGROUPTYPE_DRAW_MODAL_ALL | WM_GIZMOGROUPTYPE_3D |
             WM_GIZMOGROUPTYPE_SCALE;

  gzgt->flag = static_cast<eWM_GizmoFlagGroupTypeFlag>(flag);
  // gzgt->flag = flag;
  gzgt->gzmap_params.spaceid = SPACE_VIEW3D;
  gzgt->gzmap_params.regionid = RGN_TYPE_WINDOW;

  gzgt->poll = ED_gizmo_poll_or_unlink_delayed_from_tool;
  gzgt->setup = WIDGETGROUP_motion_curve_setup;
  gzgt->refresh = WIDGETGROUP_motion_curve_refresh;
  gzgt->draw_prepare = WIDGETGROUP_motion_curve_draw_prepare;
  gzgt->message_subscribe = WIDGETGROUP_motion_curve_message_subscribe;

  PropertyRNA *prop = RNA_def_property(gzgt->srna, "range", PROP_INT, PROP_UNSIGNED);
  RNA_def_property_ui_range(prop, 1, 10, 1, 0);
  RNA_def_property_int_default(prop, 1);
  RNA_def_property_ui_text(
      prop, "Motion Curve Range", "range of keyframes in both direction to gather anim data");
  RNA_def_property_update_runtime(prop, motion_curve_property_update);
  RNA_def_property_clear_flag(prop, PROP_ANIMATABLE);
}

static void gizmo_motion_curve_setup(struct wmGizmo *gz)
{
  WM_gizmo_set_flag(gz, WM_GIZMO_DRAW_MODAL | WM_GIZMO_EVENT_HANDLE_ALL, true);
}

static int gizmo_motion_curve_invoke(bContext *C, wmGizmo *gz, const wmEvent *event)
{
  WM_operator_name_call(C, "POSE_OT_motion_curve_dummy_for_undo", WM_OP_EXEC_DEFAULT, NULL);
  std::cout << "invoke" << std::endl;
  Scene *scene = CTX_data_scene(C);
  ARegion *ar = CTX_wm_region(C);
  MotionCurve &curve = G.curves[gz->highlight_part];

  FramePoint pt = curve.get_selected_pt(CFRA);
  G.select_pt = pt;

  bool is_check_for_outdated_pin = true;
  if (is_check_for_outdated_pin) {
    std::cout << "pins" << std::endl;
    Main *bmain = CTX_data_main(C);
    Scene *scene = CTX_data_scene(C);

    std::set<FramePoint> pins;
    for (auto pin : G.pin) {
      std::string ob_name = pin.ob_name;
      std::string pchan_name = pin.pchan_name;

      Object *ob = get_object_by_name(C, ob_name);

      bool is_pchan_exist = false;
      if (ob != NULL) {
        LISTBASE_FOREACH (bPoseChannel *, ob_pchan, &ob->pose->chanbase) {
          if (std::string(ob_pchan->name) == pchan_name) {
            is_pchan_exist = true;
            break;
          }
        }
      }

      if (ob != NULL && is_pchan_exist) {
        std::cout << pin << std::endl;
        pins.insert(pin);
      }
    }
    std::cout << "_____pin_end_____" << std::endl;
    G.pin = pins;
  }

  Object *ob = get_object_by_name(C, curve.ob_name);
  bAction *act = ob->adt->action;

  if (act != NULL) {
    if (event->shift && event->type == LEFTMOUSE) {

      MCSolver solver;
      solver.mode = 1;

      if (pt.is_keyframe()) {
        return OPERATOR_FINISHED;
      }
      else {
        int i_las_keyframe = 0;
        int i_cur_keyframe = 0;
        for (int i = 0; i < curve.pt.size(); i++) {
          if (curve.pt[i].is_keyframe()) {
            i_las_keyframe = i_cur_keyframe;
            i_cur_keyframe = i;
          }

          if (curve.pt[i_cur_keyframe].frame > pt.frame) {
            for (int x = i_las_keyframe + 1; x < i_cur_keyframe; x++) {
              float t = ((float)(x - i_las_keyframe)) / ((float)(i_cur_keyframe - i_las_keyframe));

              FramePoint &temp_pt = curve.pt[x];
              float target[3];

              float start_pos[3];
              curve.pt[i_las_keyframe].get_latest_pos(start_pos);
              float end_pos[3];
              curve.pt[i_cur_keyframe].get_latest_pos(end_pos);

              interp_v3_v3v3(target, start_pos, end_pos, t);

              solver.add_target(C, temp_pt, target);
            }
            break;
          }
        }

        for (auto &pin : G.pin) {
          solver.add_pin(C, pin, pin.cached_pos);
        }

        solver.end_add();

        ((MotionCurveItem *)gz)->solver = solver;

        return OPERATOR_RUNNING_MODAL;
      }
    }
    else if (event->ctrl && event->type == LEFTMOUSE) {
      auto iter = std::find(G.pin.begin(), G.pin.end(), pt);
      if (iter == G.pin.end()) {
        pt.get_latest_pos(pt.cached_pos);
        G.pin.insert(pt);
      }
      else {
        G.pin.erase(iter);
      }
      return OPERATOR_RUNNING_MODAL;
    }
    else if (event->alt && event->type == LEFTMOUSE) {
      float target_fra = pt.frame;
      /* set the new frame number */
      if (scene->r.flag & SCER_SHOW_SUBFRAME) {
        CFRA = (int)target_fra;
        SUBFRA = target_fra - (int)target_fra;
      }
      else {
        CFRA = round_fl_to_int(target_fra);
        SUBFRA = 0.0f;
      }
      FRAMENUMBER_MIN_CLAMP(CFRA);

      /* do updates */
      DEG_id_tag_update(&scene->id, ID_RECALC_AUDIO_SEEK);
      WM_event_add_notifier(C, NC_SCENE | ND_FRAME, scene);
    }
    else {
      if (pt.frame == CFRA) {
        MCSolver solver;
        solver.mode = 2;

        float pt_init_pos[3];
        pt.get_latest_pos(pt_init_pos);

        solver.add_target(C, pt, pt_init_pos);

        for (auto &pin : G.pin) {
          solver.add_pin(C, pin, pin.cached_pos);
        }

        solver.end_add();

        ((MotionCurveItem *)gz)->solver = solver;

        solver.solve(C);

        return OPERATOR_RUNNING_MODAL;
      }
    }
  }
  else {
    G.is_updated = false;
  }

  return OPERATOR_FINISHED;
}

static int gizmo_motion_curve_modal(bContext *C,
                                    wmGizmo *gz,
                                    const wmEvent *event,
                                    eWM_GizmoFlagTweak tweak_flag)
{
  MotionCurve &curve = G.curves[gz->highlight_part];
  if (event->ctrl) {
    auto iter = std::find(G.pin.begin(), G.pin.end(), G.select_pt);
    if (iter != G.pin.end()) {
      std::vector<FramePoint> curve_pins = curve.pt;

      for (auto &pin : curve_pins) {
        pin.get_latest_pos(pin.cached_pos);
      }

      std::set<FramePoint> new_pin;
      std::set_union(G.pin.begin(),
                     G.pin.end(),
                     curve_pins.begin(),
                     curve_pins.end(),
                     std::inserter(new_pin, new_pin.begin()));
      G.pin = new_pin;
    }
    else {
      std::set<FramePoint> new_pin;
      std::set_difference(G.pin.begin(),
                          G.pin.end(),
                          curve.pt.begin(),
                          curve.pt.end(),
                          std::inserter(new_pin, new_pin.begin()));
      G.pin = new_pin;
    }
    return OPERATOR_FINISHED;
  }
  else {

    MCSolver &solver = ((MotionCurveItem *)gz)->solver;

    if (solver.mode == 2) {
      solver.update_target_interactive(C, gz, event);
    }

    solver.solve(C);

    return OPERATOR_RUNNING_MODAL;
  }
}

static void gizmo_motion_curve_draw(const bContext *C, wmGizmo *gz)
{
  for (int i = 0; i < G.curves.size(); i++) {
    G.curves[i].is_highlight = false;
    if (gz->state & WM_GIZMO_STATE_HIGHLIGHT) {
      if (gz->highlight_part == i) {

        G.curves[i].is_highlight = true;

        // We want to also know the nearest point to mouse of on the highlighted curve
        // This logic can only be here because it is triggered on mouse move from draw_select
        // I cant put this in draw select because the region in the context in draw_select is
        // not the true region
        {
          ARegion *region = CTX_wm_region(C);
          wmWindow *win = CTX_wm_window(C);
          int i_highlight_pt = 0;
          if (win->eventstate->alt || win->eventstate->ctrl) {
            float mval_fl[2];
            // copied from eyedropper_color_sample_fl: screen space to region space
            mval_fl[0] = win->eventstate->x - region->winrct.xmin;
            mval_fl[1] = win->eventstate->y - region->winrct.ymin;

            float min_dis = std::numeric_limits<float>::max();
            float i_min_dis = -1;

            for (int ip = 0; ip < G.curves[i].pt.size(); ip++) {
              float pos[3];
              G.curves[i].pt[ip].get_latest_pos(pos);
              float co_ss[2];
              ED_view3d_project_float_global(region, pos, co_ss, V3D_PROJ_TEST_NOP);
              float dist = len_squared_v2v2(co_ss, mval_fl);
              if (dist < min_dis) {
                min_dis = dist;
                i_min_dis = ip;
              }
            }
            BLI_assert(i_min_dis > -1);
            i_highlight_pt = i_min_dis;
          }
          else {
            Scene *scene = CTX_data_scene(C);
            for (int ip = 0; ip < G.curves[i].pt.size(); ip++) {
              if (G.curves[i].pt[ip].frame == CFRA) {
                i_highlight_pt = ip;
                break;
              }
              if (G.curves[i].pt[ip].frame > CFRA) {
                BLI_assert(ip - 1 >= 0);
                i_highlight_pt = ip - 1;
                break;
              }
            }
          }

          G.curves[i].i_highlight_pt = i_highlight_pt;
        }
      }
    }

    G.curves[i].draw(-1);
  }
}

static void gizmo_motion_curve_draw_select(const bContext *C, wmGizmo *gz, int select_id)
{

  ARegion *region = CTX_wm_region(C);

  if (gz->state & WM_GIZMO_STATE_HIGHLIGHT) {
    ED_region_tag_redraw_editor_overlays(region);
  }

  for (int i = 0; i < G.curves.size(); i++) {
    G.curves[i].draw(select_id | i);
  }
}

void POSE_GT_motion_curve_item(wmGizmoType *gzt)
{
  /* identifiers */
  gzt->idname = "POSE_GT_motion_curve_item";

  /* api callbacks */
  gzt->setup = gizmo_motion_curve_setup;
  gzt->draw = gizmo_motion_curve_draw;
  gzt->draw_select = gizmo_motion_curve_draw_select;
  gzt->modal = gizmo_motion_curve_modal;
  gzt->invoke = gizmo_motion_curve_invoke;

  gzt->struct_size = sizeof(MotionCurveItem);
}

static void frame_count(struct Main * /*main*/,
                        struct PointerRNA ** /*pointers*/,
                        const int /*num_pointers*/,
                        void * /*arg*/)
{
  G.is_updated = false;
}

static bCallbackFuncStore frame_counter = {
    NULL,
    NULL,        /* next, prev */
    frame_count, /* func */
    NULL,        /* arg */
    0            /* alloc */
};

static int exec(struct bContext *, struct wmOperator *)
{
  return OPERATOR_FINISHED;
}

void POSE_OT_motion_curve_dummy_for_undo(wmOperatorType *ot)
{
  /* identifiers */
  ot->name = "Enable Motion Curve edit undo";
  ot->idname = "POSE_OT_motion_curve_dummy_for_undo";
  ot->description = "Just to make a undo push for each motion curve edit";

  ot->exec = exec;
  ot->poll = ED_operator_region_view3d_active;

  ot->flag = OPTYPE_UNDO | OPTYPE_REGISTER | OPTYPE_INTERNAL;
}

void POSE_OT_motion_curve(wmOperatorType *ot)
{
  // This is not really a operator, more like a init function.
  /* identifiers */
  ot->name = "Motion Curve register";
  ot->idname = "POSE_OT_motion_curve";
  ot->description = "motion curve of the armature";

  /* flags */

  BKE_callback_add(&frame_counter, BKE_CB_EVT_DEPSGRAPH_UPDATE_POST);
  BKE_callback_add(&frame_counter, BKE_CB_EVT_UNDO_POST);
  BKE_callback_add(&frame_counter, BKE_CB_EVT_FRAME_CHANGE_POST);

  WM_gizmogrouptype_append(POSE_GGT_motion_curve);
  WM_gizmotype_append(POSE_GT_motion_curve_item);

  StructRNA *srna = RNA_struct_find("PoseBone");

  PropertyRNA *prop;

  prop = RNA_def_property(srna, "show_head_curve", PROP_BOOLEAN, PROP_NONE);
  RNA_def_property_ui_text(prop, "Display curve for head", "N/A");
  RNA_def_property_boolean_default(prop, true);
  RNA_def_property_clear_flag(prop, PROP_ANIMATABLE);

  prop = RNA_def_property(srna, "show_tail_curve", PROP_BOOLEAN, PROP_NONE);
  RNA_def_property_ui_text(prop, "Display curve for tail", "N/A");
  RNA_def_property_boolean_default(prop, true);
  RNA_def_property_clear_flag(prop, PROP_ANIMATABLE);

  prop = RNA_def_property(srna, "use_limit_rot_x", PROP_BOOLEAN, PROP_NONE);
  RNA_def_property_ui_text(prop, "Rot X Limit", "Limit movement around the X axis");
  RNA_def_property_clear_flag(prop, PROP_ANIMATABLE);

  prop = RNA_def_property(srna, "use_limit_rot_y", PROP_BOOLEAN, PROP_NONE);
  RNA_def_property_ui_text(prop, "Rot Y Limit", "Limit movement around the Y axis");
  RNA_def_property_clear_flag(prop, PROP_ANIMATABLE);

  prop = RNA_def_property(srna, "use_limit_rot_z", PROP_BOOLEAN, PROP_NONE);
  RNA_def_property_ui_text(prop, "Rot Z Limit", "Limit movement around the Z axis");
  RNA_def_property_clear_flag(prop, PROP_ANIMATABLE);

  prop = RNA_def_property(srna, "use_limit_pos_x", PROP_BOOLEAN, PROP_NONE);
  RNA_def_property_ui_text(prop, "Pos X Limit", "Limit movement around the X axis");
  RNA_def_property_clear_flag(prop, PROP_ANIMATABLE);

  prop = RNA_def_property(srna, "use_limit_pos_y", PROP_BOOLEAN, PROP_NONE);
  RNA_def_property_ui_text(prop, "Pos Y Limit", "Limit movement around the Y axis");
  RNA_def_property_clear_flag(prop, PROP_ANIMATABLE);

  prop = RNA_def_property(srna, "use_limit_pos_z", PROP_BOOLEAN, PROP_NONE);
  RNA_def_property_ui_text(prop, "Pos Z Limit", "Limit movement around the Z axis");
  RNA_def_property_clear_flag(prop, PROP_ANIMATABLE);

  prop = RNA_def_property(srna, "ik_chain_length", PROP_INT, PROP_UNSIGNED);
  RNA_def_property_ui_range(prop, 0, 5, 1, 0);
  RNA_def_property_int_default(prop, 0);
  RNA_def_property_ui_text(
      prop, "IK chain length for adjustment", "Limiting the number of bones involved");
  RNA_def_property_clear_flag(prop, PROP_ANIMATABLE);
}

/****** Helper functions *******/

Object *get_object_by_name(bContext *C, std::string name)
{
  Scene *scene = CTX_data_scene(C);
  return BKE_scene_object_find_by_name(scene, name.c_str() + 2);
}

/* This function is simply copied from existing anim path implementation*/
void DEG_update(Depsgraph *depsgraph, Main *bmain)
{
  Scene *scene = DEG_get_input_scene(depsgraph);
  ViewLayer *view_layer = DEG_get_input_view_layer(depsgraph);

  /* Keep this first. */
  BKE_callback_exec_id(bmain, &scene->id, BKE_CB_EVT_FRAME_CHANGE_PRE);

  for (int pass = 0; pass < 2; pass++) {
    /* Update animated image textures for particles, modifiers, gpu, etc,
     * call this at the start so modifiers with textures don't lag 1 frame.
     */
    DEG_graph_relations_update(depsgraph, bmain, scene, view_layer);
#ifdef POSE_ANIMATION_WORKAROUND
    scene_armature_depsgraph_workaround(bmain, depsgraph);
#endif
    /* Update all objects: drivers, matrices, displists, etc. flags set
     * by depgraph or manual, no layer check here, gets correct flushed.
     *
     * NOTE: Only update for new frame on first iteration. Second iteration is for ensuring
     * user edits from callback are properly taken into account. Doing a time update on those
     * would loose any possible unkeyed changes made by the handler. */
    if (pass == 0) {
      const float ctime = BKE_scene_frame_get(scene);
      DEG_evaluate_on_framechange(bmain, depsgraph, ctime);
    }
    else {
      DEG_evaluate_on_refresh(bmain, depsgraph);
    }

    /* Notify editors and python about recalc. */
    if (pass == 0) {
      BKE_callback_exec_id_depsgraph(bmain, &scene->id, depsgraph, BKE_CB_EVT_FRAME_CHANGE_POST);
    }

    /* Inform editors about possible changes. */
    // DEG_ids_check_recalc(bmain, depsgraph, scene, view_layer, true);
    /* clear recalc flags */
    DEG_ids_clear_recalc(bmain, depsgraph);

    /* If user callback did not tag anything for update we can skip second iteration.
     * Otherwise we update scene once again, but without running callbacks to bring
     * scene to a fully evaluated state with user modifications taken into account. */
    if (DEG_is_fully_evaluated(depsgraph)) {
      break;
    }
  }
}

struct bToolRef *WM_toolsystem_ref_from_const_context(const struct bContext *C)
{
  WorkSpace *workspace = CTX_wm_workspace(C);
  ViewLayer *view_layer = CTX_data_view_layer(C);
  ScrArea *sa = CTX_wm_area(C);
  if (((1 << sa->spacetype) & WM_TOOLSYSTEM_SPACE_MASK) == 0) {
    return NULL;
  }
  const bToolKey tkey = {
      sa->spacetype,
      WM_toolsystem_mode_from_spacetype(view_layer, sa, sa->spacetype),
  };
  bToolRef *tref = WM_toolsystem_ref_find(workspace, &tkey);
  /* We could return 'sa->runtime.tool' in this case. */
  if (sa->runtime.is_tool_set) {
    BLI_assert(tref == sa->runtime.tool);
  }
  return tref;
}

bool my_quadprog(const Eigen::MatrixXd &Q,
                        const Eigen::VectorXd &c,
                        const std::vector<Eigen::Matrix3Xd> &Jc,
                        const std::vector<Eigen::Vector3d> &dP,
                        Eigen::VectorXd &x)
{
  // The problem is in the form:
  //
  // min 0.5 * x Q x + c x
  // subject to
  // (dP-Jc * x) ^ 2 < 0.001

  int num_parm = x.rows();
  double t = 1;
  int num_cons = Jc.size();
  Eigen::VectorXd x_old = x;
  double near_zero = 0.0001;
  Eigen::MatrixXd reg = 0.0000001 * Eigen::MatrixXd::Identity(num_parm, num_parm);
  int dbg_ctr = 0;
  bool is_feasible = false;

  if (num_cons == 0) {
    is_feasible = true;
  }
  else {
    dbg_ctr = 0;
    const int max_trials = 5;
    while (!is_feasible && dbg_ctr < max_trials) {
      Eigen::MatrixXd H = Eigen::MatrixXd::Zero(num_parm, num_parm);
      Eigen::VectorXd G = Eigen::VectorXd::Zero(num_parm);
      // gradient of the quadratic energy

      std::vector<Eigen::VectorXd> cons_grads(num_cons);

      is_feasible = true;
      for (int i = 0; i < num_cons; i++) {
        if ((dP[i] - Jc[i] * x).squaredNorm() - near_zero / 10 >= 0) {
          is_feasible = false;
          break;
        }
      }

      if (is_feasible == false) {
        for (int i = 0; i < num_cons; i++) {
          cons_grads[i] = 2 * (Jc[i].transpose() * Jc[i] + reg) * x -
                          2 * Jc[i].transpose() * dP[i];

          G = G + cons_grads[i];
        }

        for (int i = 0; i < num_cons; i++) {

          Eigen::MatrixXd Hess_i = 2 * (Jc[i].transpose() * Jc[i] + reg);

          H = H + Hess_i;
        }

        Eigen::LLT<Eigen::MatrixXd> H_(H);

        Eigen::VectorXd step = -H_.solve(G);

        x = x + step;
        dbg_ctr++;
      }
    }
  }

  if (!is_feasible) {
    std::cout << "Could not recover from violated constraint" << std::endl;
    x.setZero();
  }
  else {
    dbg_ctr = 0;
    while (dbg_ctr < 1) {

      Eigen::MatrixXd H = Eigen::MatrixXd::Zero(num_parm, num_parm);
      Eigen::VectorXd G = Eigen::VectorXd::Zero(num_parm);

      // gradient of the quadratic energy
      G = t * Q * x + t * c;

      std::vector<Eigen::VectorXd> cons_grads(num_cons);
      std::vector<double> cons_value(num_cons);

      for (int i = 0; i < num_cons; i++) {

        // -1 / fi(x) * grad(fi(x))
        cons_value[i] = (dP[i] - Jc[i] * x).squaredNorm() - near_zero;

        if (cons_value[i] > 0) {
          std::cout << "Detected constraint violation!" << std::endl;
          x = x_old;
          return false;
        }

        cons_grads[i] = -1.0 / cons_value[i] *
                        (2 * (Jc[i].transpose() * Jc[i] + reg) * x -
                         2 * Jc[i].transpose() * dP[i]);

        G = G + cons_grads[i];
      }

      H = t * Q;

      for (int i = 0; i < num_cons; i++) {

        Eigen::MatrixXd Hess_i = 1.0 / square_d(cons_value[i]) * cons_grads[i] *
                                     cons_grads[i].transpose() +
                                 (-1.0) / cons_value[i] * (2 * (Jc[i].transpose() * Jc[i] + reg));

        H = H + Hess_i;
      }

      Eigen::LLT<Eigen::MatrixXd> H_(H);

      Eigen::VectorXd step = -H_.solve(G);

      double step_size = 1;

      x_old = x;
      x = x + step_size * step;

      dbg_ctr++;
    }
  }
  return false;
}

void get_fcurve_segment_ex(
    std::vector<FCurveSegment> &segs, CurveType type, Object *ob, bPoseChannel *pchan, float frame)
{
  bAction *act = ob->adt->action;
  PointerRNA ptr;
  RNA_pointer_create((ID *)ob, &RNA_PoseBone, pchan, &ptr);
  char *basePath = RNA_path_from_ID_to_struct(&ptr);

  ListBase curve = {NULL, NULL};
  action_get_item_transforms(act, ob, pchan, &curve);

  bool use_limit_rot_x = RNA_boolean_get(&ptr, "use_limit_rot_x");
  bool use_limit_rot_y = RNA_boolean_get(&ptr, "use_limit_rot_y");
  bool use_limit_rot_z = RNA_boolean_get(&ptr, "use_limit_rot_z");

  bool use_limit_pos_x = RNA_boolean_get(&ptr, "use_limit_pos_x");
  bool use_limit_pos_y = RNA_boolean_get(&ptr, "use_limit_pos_y");
  bool use_limit_pos_z = RNA_boolean_get(&ptr, "use_limit_pos_z");

  LISTBASE_FOREACH (LinkData *, link, &curve) {
    FCurve *fcu = (FCurve *)link->data;
    const char *bPtr = NULL, *pPtr = NULL;
    bPtr = strstr(fcu->rna_path, basePath);
    bPtr += strlen(basePath);

    pPtr = strstr(bPtr, CURVE_TYPE_NAME[type]);

    if (pPtr) {
      if (type == ROT_EUL) {
        if (fcu->array_index == 0 && use_limit_rot_x) {
          continue;
        }
        if (fcu->array_index == 1 && use_limit_rot_y) {
          continue;
        }
        if (fcu->array_index == 2 && use_limit_rot_z) {
          continue;
        }
      }

      if (type == LOC) {
        if (fcu->array_index == 0 && use_limit_pos_x) {
          continue;
        }
        if (fcu->array_index == 1 && use_limit_pos_y) {
          continue;
        }
        if (fcu->array_index == 2 && use_limit_pos_z) {
          continue;
        }
      }

      int i = 0;
      for (i = 0; i < fcu->totvert; i++) {
        if (fcu->bezt[i].vec[1][0] == frame) {
          // There is a keyframe at this frame
          FCurveSegment seg;
          seg.ob_name = ob->id.name;
          seg.pchan_name = pchan->name;
          seg.fcu = fcu;
          seg.type = type;
          seg.keyframe_idx.push_back(i);
          segs.push_back(seg);
          break;
        }
        else if (fcu->bezt[i].vec[1][0] > frame) {
          // There is a keyframe to the right of this frame
          if (i > 0) {
            // And if there is also a keyframe on the left side
            FCurveSegment seg;
            seg.ob_name = ob->id.name;
            seg.pchan_name = pchan->name;
            seg.fcu = fcu;
            seg.type = type;
            seg.keyframe_idx.push_back(i - 1);
            seg.keyframe_idx.push_back(i);
            segs.push_back(seg);
          }
          break;
        }
      }
    }
  }

  MEM_freeN(basePath);
  BLI_freelistN(&curve);
}

void get_sorted_fcurve_segment(bContext *C, std::vector<FCurveSegment> &segs, FramePoint pt)
{
  Scene *scene = CTX_data_scene(C);

  Object *ob = get_object_by_name(C, pt.ob_name);

  bPoseChannel *pchan = BKE_pose_channel_find_name(ob->pose, pt.pchan_name.c_str());

  float frame = pt.frame;
  FramePointType pt_type = pt.pt_type;

  if (pt_type == TAIL) {
    switch (pchan->rotmode) {
      case ROT_MODE_XYZ:
      case ROT_MODE_XZY:
      case ROT_MODE_YXZ:
      case ROT_MODE_YZX:
      case ROT_MODE_ZXY:
      case ROT_MODE_ZYX:
        get_fcurve_segment_ex(segs, ROT_EUL, ob, pchan, frame);
        break;
    }
  }

  if ((pchan->bone->flag & BONE_CONNECTED) != BONE_CONNECTED) {
    get_fcurve_segment_ex(segs, LOC, ob, pchan, frame);
  }

  bPoseChannel *chain_pchan = pchan->parent;

  while (chain_pchan != NULL) {

    switch (chain_pchan->rotmode) {
      case ROT_MODE_XYZ:
      case ROT_MODE_XZY:
      case ROT_MODE_YXZ:
      case ROT_MODE_YZX:
      case ROT_MODE_ZXY:
      case ROT_MODE_ZYX:
        get_fcurve_segment_ex(segs, ROT_EUL, ob, chain_pchan, frame);
        break;
    }
    if ((chain_pchan->bone->flag & BONE_CONNECTED) != BONE_CONNECTED) {
      get_fcurve_segment_ex(segs, LOC, ob, chain_pchan, frame);
    }

    chain_pchan = chain_pchan->parent;
  }

  std::sort(segs.begin(), segs.end());
}

void get_sorted_primary_segments(bContext *C, std::vector<FCurveSegment> &segs, FramePoint pt)
{

  Scene *scene = CTX_data_scene(C);

  Object *ob = get_object_by_name(C, pt.ob_name);

  bPoseChannel *pchan = BKE_pose_channel_find_name(ob->pose, pt.pchan_name.c_str());
  PointerRNA ptr = {0};
  RNA_pointer_create((ID *)ob, &RNA_PoseBone, pchan, &ptr);
  int depth_limit = RNA_int_get(&ptr, "ik_chain_length");

  float frame = pt.frame;
  FramePointType pt_type = pt.pt_type;

  if (pt_type == HEAD) {
    if ((pchan->bone->flag & BONE_CONNECTED) != BONE_CONNECTED) {
      get_fcurve_segment_ex(segs, LOC, ob, pchan, frame);
    }
  }
  else {
    bPoseChannel *chain_pchan = pchan;
    int chain_depth = 0;
    while (chain_pchan != NULL) {

      switch (chain_pchan->rotmode) {
        case ROT_MODE_XYZ:
        case ROT_MODE_XZY:
        case ROT_MODE_YXZ:
        case ROT_MODE_YZX:
        case ROT_MODE_ZXY:
        case ROT_MODE_ZYX:
          get_fcurve_segment_ex(segs, ROT_EUL, ob, chain_pchan, frame);
          break;
      }

      chain_pchan = chain_pchan->parent;
      chain_depth += 1;

      if (depth_limit != 0 && chain_depth >= depth_limit) {
        break;
      }
    }
  }

  std::sort(segs.begin(), segs.end());
}
