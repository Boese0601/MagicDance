include "base.thrift"
include "common.thrift"

namespace cpp faceattr
namespace go faceattr
namespace py faceattr


struct FaceAttrReq{
  1: common.BaseInfo base_info,                       //基本的图像输入
  2: optional map<string, double> configs,            //配置信息
  255:optional base.Base Base,
}

struct FaceAttrVideoReq{
  1: common.VideoBaseInfo base_info,                       //基本的图像输入
  2: optional map<string, double> configs,            //配置信息
  255:optional base.Base Base,
}

struct FaceAttrInfo{
  1:common.Rect rect,
  2:list<common.Point> landmark106s,
  3:map<string, double> attrs,
  248:optional list<common.Point> landmark_outlines,
  249:optional list<common.Point> landmark_eye_lefts,
  250:optional list<common.Point> landmark_eye_rights,
  251:optional list<common.Point> landmark_eyebrow_lefts,
  252:optional list<common.Point> landmark_eyebrow_rights,
  253:optional list<common.Point> landmark_lips,
  254:optional list<common.Point> landmark_iris_lefts,
  255:optional list<common.Point> landmark_iris_rights,
}

struct FaceAttrRsp {
  1:string status,
  2:list<FaceAttrInfo> face_attr_infos,
  3:string version = "1.0.0",
  254:optional common.BaseMeta base_meta,
  255:optional base.BaseResp BaseResp,
}

struct FaceAttrVideoRsp {
  1:string status,
  2:list<FaceAttrRsp> frame_rsps,
  3:string version = "1.0.0",
  254:optional common.BaseMeta base_meta,
  255:optional base.BaseResp BaseResp,
}

service FaceAttr {
  FaceAttrRsp PredictFaceAttr(1: FaceAttrReq req)
}
