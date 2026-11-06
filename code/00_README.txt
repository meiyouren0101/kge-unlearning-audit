包 A v3：FB / WN 20%，按种子拆成六条链（仅调度变更）

FB种子0/42/123、WN种子0/42/123，共六条链。--workers可选1至6，默认6。
每条链按请求0→1→2→3串行；请求内保留v2原来的方法顺序。stage名称、
全部科学输入、模型加载、批大小、优化器、种子、评价与输出布局不变。
每图仍337项：seed0有113项，seed42/123各112项；两图合计674项。
每图只有seed0执行一次scale_ZF，另外两链等它的commit.json后读取同一eta。
workers=1也不会因网格生产者排在消费者后面而死锁。全部六链结束后才统一
进行FB、WN的CPU审计；audit_session函数本身的原始字节与v2相同。

并行与资源规则
每条链的科学worker是独立进程，保留原RNG初始化和数据/检查点读取路径。
所有启动共用 /mnt/new4T/lsb/.graphdpo_gpu0_admission.lock：flock锁内查显存，
锁内启动，45秒后释放。显存不足释放锁、60秒重查；依赖300秒重查不持锁。
没有新增每进程显存上限或torch内存比例设置。workers=6是上限，不是同时
占满显卡的保证；16GiB是否能容纳六项取决于当时阶段的真实显存占用。
准入门槛、OOM判定和额度、默认7200秒、Fisher删除6/12/24小时均与v2相同。
一条链技术故障会取消本包其他链，仅终止本包启动的worker，保留全部现场。
每链的状态与峰值历史在 FB或WN/chains/seed_<seed>/ 下，各自写入避免冲突。
progress同时显示六链封口数、当前项、等待原因、心跳与同类实测剩余时间。
尚无同类计时的阶段列为unknown，不把部分已知小时当成全程剩余时间。

安全替换v2
必须由用户先确认旧v2无已封口阶段、正在等待、无worker或其他子进程。
safe_stop_v2.py默认只读检查；--stop才会执行安全停止。停止工具先持有同一
共享启动锁，防止检查后v2突然开工；验证进程路径、UID、PID起始时间，然后
用pidfd只向v2的job20.py run协调器发送SIGTERM。等待退出、确认v2无进程且
其coordinator锁已释放，最后释放共享锁。在v3/switch_receipts写停止回执。
工具不删除v2任何文件、不删除锁文件、不发SIGKILL、不杀其他作业。v2可能
自行在旧目录写下正常的中止记录，这是保留的现场。若阶段已开始或已有
封口，它会拒绝此零阶段切换；此时不要运行v3，先贴出检查输出。
旧v2结果/重试/状态保留；v3使用独立新目录/new results，没有共享旧状态。
v3启动前另检查精确v2路径是否仍有活进程；不会误匹配B、YAGO或ZeroFisher。
服务器需要Linux pidfd接口（Ubuntu/Python3.10已有）；本地测试不冒称Linux实测。

操作入口（完整上传/部署/停止/启动命令另见交付文件）
export GPU_PYTHON=/home/lsb/work/asd_kd_pilot_v2_job/.venv/bin/python
bash check.sh "$GPU_PYTHON" /home/lsb/work/fb15k-237-20.zip /home/lsb/work/wn18rr-20.zip
python3 -I -B safe_stop_v2.py --v2-root /home/lsb/work/graphdpo_fb_wn20_full_baselines_v2_job
python3 -I -B safe_stop_v2.py --v2-root /home/lsb/work/graphdpo_fb_wn20_full_baselines_v2_job --stop
bash launch.sh all --trust-checkpoint --workers 6 --eta-policy grid --fb-zip /home/lsb/work/fb15k-237-20.zip --wn-zip /home/lsb/work/wn18rr-20.zip
bash progress.sh
bash report.sh
启动为nohup+setsid，SSH断开可继续；不自动跨服务器重启。不要重复启动。
报告ZIP仍在results/run_*/report_*.zip，不含模型/排名缓存/数据。

证明：validation/V3_SCIENCE_COMPARE.json列出156个未改科学/配置文件的
v2/v3 SHA256，另列嵌入job20.py的audit_session原始字节SHA256。
validation/V3_FILE_COMPARISON.csv逐文件列出整个v2→v3变动；diff只涉及调度、
状态路径、检查/安全切换、新测试及文档。本轮不运行真实服务器或GPU实验。
