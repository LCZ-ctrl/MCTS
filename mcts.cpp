#include<iostream>
#include<string>
#include<cstdlib>
#include<ctime>
#include<vector>
#include<cmath>
#include<chrono>

#define GRIDSIZE 8  // 定义棋盘大小
#define OBSTACLE 2  // 定义障碍物的标记
#define judge_black 0  // 黑方标记
#define judge_white 1  // 白方标记
#define grid_black 1  // 黑棋在棋盘中的标记
#define grid_white -1  // 白棋在棋盘中的标记

using namespace std;
using namespace std::chrono;

int currBotColor;  // 当前下棋的颜色
int gridInfo[GRIDSIZE][GRIDSIZE] = { 0 };  // 初始化棋盘状态，0代表空，1代表黑棋，-1代表白棋
int dx[] = { -1,-1,-1,0,0,1,1,1 };  // x方向的步伐
int dy[] = { -1,0,1,-1,1,-1,0,1 };  // y方向的步伐

// 判断坐标是否在棋盘内
inline bool inMap(int x, int y) {
	return x >= 0 && x < GRIDSIZE && y >= 0 && y < GRIDSIZE;
}

// 在坐标(x0, y0)到(x1, y1)之间模拟走子，并放置障碍物(x2, y2)
bool ProcStep(int x0, int y0, int x1, int y1, int x2, int y2, int color, bool check_only) {
	// 检查坐标是否越界
	if (!inMap(x0, y0) || !inMap(x1, y1) || !inMap(x2, y2)) {
		return false;
	}
	// 检查是否为该玩家的棋子，目标位置是否为空，障碍物位置是否合适
	if (gridInfo[x0][y0] != color || gridInfo[x1][y1] != 0) {
		return false;
	}
	if (gridInfo[x2][y2] != 0 && !(x2 == x0 && y2 == y0)) {
		return false;
	}

	if (!check_only) {
		gridInfo[x0][y0] = 0;  // 清空原位置
		gridInfo[x1][y1] = color;  // 放置棋子
		gridInfo[x2][y2] = OBSTACLE;  // 放置障碍物
	}
	return true;
}

// 定义一个结构体Move，用来表示一次棋盘上的动作（起点、终点和障碍物位置）
struct Move {
	int x0, y0, x1, y1, x2, y2;  // 起点、终点、障碍物
};

// 定义一个结构体State，用来表示一个棋盘局面
struct State {
	int board[GRIDSIZE][GRIDSIZE];  // 棋盘状态
	int player;  // 当前下棋的颜色，1为黑方，-1为白方

	// 获取当前局面下所有合法的走法
	void getMoves(vector<Move>& moves) const {
		moves.clear();  // 清空所有走法
		// 遍历棋盘，找到当前玩家的棋子
		for (int i = 0; i < GRIDSIZE; ++i) {
			for (int j = 0; j < GRIDSIZE; ++j) {
				if (board[i][j] != player) continue;  // 如果当前位置不是当前玩家的棋子，跳过

				// 尝试所有8个方向的走法
				for (int d1 = 0; d1 < 8; ++d1) {
					int max_step = 0;
					int x = i, y = j;
					// 向该方向走，直到遇到障碍物或越界
					while (true) {
						x += dx[d1];
						y += dy[d1];
						if (!inMap(x, y) || board[x][y] != 0) break;  // 如果越界或遇到非空格，停止
						max_step++;  // 记录最大步数
					}

					// 尝试不同步数的移动
					for (int s = 1; s <= max_step; ++s) {
						int ni = i + dx[d1] * s;  // 新的棋子位置
						int nj = j + dy[d1] * s;

						// 尝试在目标位置放置障碍物
						for (int d2 = 0; d2 < 8; ++d2) {
							int x_ob = ni, y_ob = nj;
							int step = 0;
							while (true) {
								x_ob += dx[d2];
								y_ob += dy[d2];
								if (!inMap(x_ob, y_ob) || (board[x_ob][y_ob] != 0 && !(x_ob == i && y_ob == j)))
									break;
								step++;
							}
							for (int s_ob = 1; s_ob <= step; ++s_ob) {
								int oi = ni + dx[d2] * s_ob;
								int oj = nj + dy[d2] * s_ob;
								moves.push_back({ i, j, ni, nj, oi, oj });
							}
						}
					}
				}
			}
		}
	}

	// 进行一次走棋操作
	void doMove(const Move& m) {
		board[m.x0][m.y0] = 0;  // 清空原位置
		board[m.x1][m.y1] = player;  // 放置到新的位置
		board[m.x2][m.y2] = OBSTACLE;  // 放置障碍物
		player = -player;  // 切换玩家
	}

	// 判断当前局面是否为终局
	bool isTerminal() const {
		vector<Move> tmp;
		getMoves(tmp);  // 获取当前局面的合法走法
		return tmp.empty();  // 如果没有合法走法，则为终局
	}

	// 随机模拟直到终局，返回结果
	int rollout(int root) const {
		State s = *this;  // 复制当前局面
		vector<Move> m;
		const int MAX_STEPS = 200;  // 最大模拟步数
		int steps = 0;

		while (steps++ < MAX_STEPS) {
			s.getMoves(m);  // 获取合法走法
			if (m.empty()) return (s.player == root) ? -1 : 1;  // 无路可走时，返回胜负
			s.doMove(m[rand() % m.size()]);  // 随机选择一走法
			m.clear();
		}
		return 0;  // 模拟超过最大步数仍未结束，返回0
	}
};

// MCTS树节点结构
struct Node {
	State state;  // 当前局面
	Move move;  // 从父节点到当前节点的走法
	Node* parent;  // 父节点指针
	vector<Node*> children;  // 子节点列表
	vector<Move> untriedMoves;  // 未尝试的走法
	int visits;  // 当前节点被访问次数
	double value;  // 当前节点的累计价值（用于 UCB）

	// 构造函数
	Node(const State& st, Node* p = nullptr, Move m = Move{ -1,-1,-1,-1,-1,-1 })
		: state(st), move(m), parent(p), visits(0), value(0.0) {
		state.getMoves(untriedMoves);  // 生成所有的合法走法
	}

	// 根据UCB选择最佳子节点
	Node* selectChild() {
		Node* best = nullptr;
		double bestU = -1e18;
		// 遍历所有子节点，选择UCB值最大的子节点
		for (size_t i = 0; i < children.size(); ++i) {
			Node* c = children[i];
			double u = c->value / c->visits + 1.414 * sqrt(log(visits) / c->visits);  // UCB公式
			if (u > bestU) {
				bestU = u;
				best = c;
			}
		}
		return best;  // 返回最佳子节点
	}

	// 从未尝试的走法中选一个，生成新的子节点
	Node* expand() {
		Move m = untriedMoves.back();  // 选择最后一个未尝试的走法
		untriedMoves.pop_back();  // 移除该走法
		State next = state;
		next.doMove(m);  // 进行走棋
		Node* child = new Node(next, this, m);  // 创建新子节点
		children.push_back(child);  // 将子节点添加到子节点列表
		return child;
	}

	// 回传模拟结果，更新当前节点的值
	void backup(double result) {
		visits++;  // 访问次数加1
		value += result;  // 累加模拟结果
		if (parent) parent->backup(-result);  // 反向更新父节点的结果
	}
};

int main() {
	srand(time(NULL));  // 初始化随机数生成器
	int x0, y0, x1, y1, x2, y2;

	// 初始化棋盘状态
	gridInfo[0][2] = grid_black;
	gridInfo[2][0] = grid_black;
	gridInfo[5][0] = grid_black;
	gridInfo[7][2] = grid_black;
	gridInfo[0][5] = grid_white;
	gridInfo[2][7] = grid_white;
	gridInfo[5][7] = grid_white;
	gridInfo[7][5] = grid_white;

	// 输入回合数
	int turnID;
	cin >> turnID;

	// 假设开始时是白方
	currBotColor = grid_white;
	for (int i = 0; i < turnID; ++i) {
		cin >> x0 >> y0 >> x1 >> y1 >> x2 >> y2;
		if (x0 == -1) currBotColor = grid_black;  // 如果对方先手
		else ProcStep(x0, y0, x1, y1, x2, y2, -currBotColor, false);  // 模拟对方落子

		if (i < turnID - 1) {
			cin >> x0 >> y0 >> x1 >> y1 >> x2 >> y2;
			if (x0 >= 0) ProcStep(x0, y0, x1, y1, x2, y2, currBotColor, false);  // 模拟本方落子
		}
	}

	// 初始化根节点的局面
	State rootState;
	for (int i = 0; i < GRIDSIZE; ++i)
		for (int j = 0; j < GRIDSIZE; ++j)
			rootState.board[i][j] = gridInfo[i][j];
	rootState.player = currBotColor;

	// 创建根节点
	Node* root = new Node(rootState);
	auto start = high_resolution_clock::now();  // 记录起始时间
	double elapsed = 0.0;

	// MCTS主循环
	while (elapsed < 0.9) {  // 限制时间为0.9秒
		Node* node = root;

		// 选择和扩展阶段
		while (node->untriedMoves.empty() && !node->children.empty() && !node->state.isTerminal()) {
			node = node->selectChild();  // 选择UCB值最大的子节点
		}

		// 扩展一个子节点
		if (!node->untriedMoves.empty()) {
			node = node->expand();
		}

		// 随机模拟
		int result = node->state.rollout(currBotColor);
		node->backup(result);  // 回传模拟结果

		// 更新已用时间
		auto now = high_resolution_clock::now();
		elapsed = duration<double>(now - start).count();
	}

	// 选择访问次数最多的子节点
	Node* bestChild = nullptr;
	int maxVisits = -1;
	for (size_t i = 0; i < root->children.size(); ++i) {
		if (root->children[i]->visits > maxVisits) {
			maxVisits = root->children[i]->visits;
			bestChild = root->children[i];
		}
	}

	// 输出最优走法
	if (bestChild) {
		Move m = bestChild->move;
		cout << m.x0 << " " << m.y0 << " " << m.x1 << " " << m.y1 << " " << m.x2 << " " << m.y2 << endl;
	}
	else {
		cout << "-1 -1 -1 -1 -1 -1" << endl;  // 如果没有可行的走法
	}

	return 0;
}