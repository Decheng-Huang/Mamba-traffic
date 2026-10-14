#!/usr/bin/env bash
# box_fetch.sh —— 把重跑的结果、出生证明、日志打成一个包，供下载。
#
# 跑法：  bash /root/autodl-tmp/box_fetch.sh
# 然后在 JupyterLab 文件树里找到 /root/autodl-tmp/prov_out.tgz → 右键 Download。
#
# ★ 规矩（证据链说明.md §六）：**绝不让唯一副本只存在于实例上。**
#   实例一释放，盘上什么都不剩 —— 跑完立刻取回。
set -u
BASE=/root/autodl-tmp
cd "$BASE" || { echo "FATAL: cd $BASE 失败"; exit 1; }

# 逐个点名；不在场的跳过（不报错），在场的必须进包。
# 不写死 tar 成员表，是因为重跑可能只跑完前几组就被 Ctrl-C ——
# 那种情况下部分结果仍然有效，也要能取回来。
FILES=""
for f in paper_results_reprov.json paper_results_smoke.json \
         _provenance reprov_log smoke.log reprov_driver.log; do
  if [ -e "$f" ]; then FILES="$FILES $f"; else echo "  （跳过，不在场）$f"; fi
done

if [ -z "$FILES" ]; then
  echo "★ 没有任何可取的产物 —— 重跑是不是还没跑？"
  exit 1
fi

echo "打包：$FILES"
rm -f prov_out.tgz
# shellcheck disable=SC2086
tar czf prov_out.tgz $FILES

echo
echo "== 包内容 =="
tar tzf prov_out.tgz
echo
ls -l prov_out.tgz
echo "sha256（取回后本地要对得上）："
sha256sum prov_out.tgz
echo
echo "★ 现在去 JupyterLab 下载 $BASE/prov_out.tgz"
