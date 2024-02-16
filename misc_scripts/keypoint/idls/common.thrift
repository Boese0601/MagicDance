include "base.thrift"
namespace py common
namespace go common
namespace cpp common

struct Rect{
  1: double left,
  2: double top,
  3: double right,
  4: double bottom,
}

struct Point{
  1: double x,
  2: double y
  3: bool status,
}

struct BaseInfo{
  1: string image_name,
  2: binary image_data,
  254:string caller="default",
}

struct VideoBaseInfo{
  1: string video_name,
  2: optional binary video_data,
  3: optional string video_url,    # url or binary 二者至少提供一个
  4: i32 max_frame = -1,                # 指定抽帧的帧数(max_frame 和frame_interval 只能二选一)
  5: i32 frame_interval = -1,            # 帧采样间隔，每秒采集一帧，那就是25
  254:string caller="default",
}

struct BaseMeta{
  1: i32 image_width,
  2: i32 image_height,
}

struct VideoMQRsp{
  1: string req_id,
  2: string rsp_url,
  3: string video_url = "",
  4: i32 status,
  5: string message,
}
