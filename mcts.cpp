#include<iostream>
#include<string>
#include<cstdlib>
#include<ctime>
#include<vector>
#include<cmath>
#include<chrono>

#define GRIDSIZE 8  // �������̴�С
#define OBSTACLE 2  // �����ϰ���ı��
#define judge_black 0  // �ڷ����
#define judge_white 1  // �׷����
#define grid_black 1  // �����������еı��
#define grid_white -1  // �����������еı��

using namespace std;
using namespace std::chrono;

int currBotColor;  // ��ǰ�������ɫ
int gridInfo[GRIDSIZE][GRIDSIZE] = { 0 };  // ��ʼ������״̬��0����գ�1������壬-1�������
int dx[] = { -1,-1,-1,0,0,1,1,1 };  // x����Ĳ���
int dy[] = { -1,0,1,-1,1,-1,0,1 };  // y����Ĳ���

// �ж������Ƿ���������
inline bool inMap(int x, int y) {
	return x >= 0 && x < GRIDSIZE && y >= 0 && y < GRIDSIZE;
}

// ������(x0, y0)��(x1, y1)֮��ģ�����ӣ��������ϰ���(x2, y2)
bool ProcStep(int x0, int y0, int x1, int y1, int x2, int y2, int color, bool check_only) {
	// ��������Ƿ�Խ��
	if (!inMap(x0, y0) || !inMap(x1, y1) || !inMap(x2, y2)) {
		return false;
	}
	// ����Ƿ�Ϊ����ҵ����ӣ�Ŀ��λ���Ƿ�Ϊ�գ��ϰ���λ���Ƿ����
	if (gridInfo[x0][y0] != color || gridInfo[x1][y1] != 0) {
		return false;
	}
	if (gridInfo[x2][y2] != 0 && !(x2 == x0 && y2 == y0)) {
		return false;
	}

	if (!check_only) {
		gridInfo[x0][y0] = 0;  // ���ԭλ��
		gridInfo[x1][y1] = color;  // ��������
		gridInfo[x2][y2] = OBSTACLE;  // �����ϰ���
	}
	return true;
}

// ����һ���ṹ��Move��������ʾһ�������ϵĶ�������㡢�յ���ϰ���λ�ã�
struct Move {
	int x0, y0, x1, y1, x2, y2;  // ��㡢�յ㡢�ϰ���
};

// ����һ���ṹ��State��������ʾһ�����̾���
struct State {
	int board[GRIDSIZE][GRIDSIZE];  // ����״̬
	int player;  // ��ǰ�������ɫ��1Ϊ�ڷ���-1Ϊ�׷�

	// ��ȡ��ǰ���������кϷ����߷�
	void getMoves(vector<Move>& moves) const {
		moves.clear();  // ��������߷�
		// �������̣��ҵ���ǰ��ҵ�����
		for (int i = 0; i < GRIDSIZE; ++i) {
			for (int j = 0; j < GRIDSIZE; ++j) {
				if (board[i][j] != player) continue;  // �����ǰλ�ò��ǵ�ǰ��ҵ����ӣ�����

				// ��������8��������߷�
				for (int d1 = 0; d1 < 8; ++d1) {
					int max_step = 0;
					int x = i, y = j;
					// ��÷����ߣ�ֱ�������ϰ����Խ��
					while (true) {
						x += dx[d1];
						y += dy[d1];
						if (!inMap(x, y) || board[x][y] != 0) break;  // ���Խ��������ǿո�ֹͣ
						max_step++;  // ��¼�����
					}

					// ���Բ�ͬ�������ƶ�
					for (int s = 1; s <= max_step; ++s) {
						int ni = i + dx[d1] * s;  // �µ�����λ��
						int nj = j + dy[d1] * s;

						// ������Ŀ��λ�÷����ϰ���
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

	// ����һ���������
	void doMove(const Move& m) {
		board[m.x0][m.y0] = 0;  // ���ԭλ��
		board[m.x1][m.y1] = player;  // ���õ��µ�λ��
		board[m.x2][m.y2] = OBSTACLE;  // �����ϰ���
		player = -player;  // �л����
	}

	// �жϵ�ǰ�����Ƿ�Ϊ�վ�
	bool isTerminal() const {
		vector<Move> tmp;
		getMoves(tmp);  // ��ȡ��ǰ����ĺϷ��߷�
		return tmp.empty();  // ���û�кϷ��߷�����Ϊ�վ�
	}

	// ���ģ��ֱ���վ֣����ؽ��
	int rollout(int root) const {
		State s = *this;  // ���Ƶ�ǰ����
		vector<Move> m;
		const int MAX_STEPS = 200;  // ���ģ�ⲽ��
		int steps = 0;

		while (steps++ < MAX_STEPS) {
			s.getMoves(m);  // ��ȡ�Ϸ��߷�
			if (m.empty()) return (s.player == root) ? -1 : 1;  // ��·����ʱ������ʤ��
			s.doMove(m[rand() % m.size()]);  // ���ѡ��һ�߷�
			m.clear();
		}
		return 0;  // ģ�ⳬ���������δ����������0
	}
};

// MCTS���ڵ�ṹ
struct Node {
	State state;  // ��ǰ����
	Move move;  // �Ӹ��ڵ㵽��ǰ�ڵ���߷�
	Node* parent;  // ���ڵ�ָ��
	vector<Node*> children;  // �ӽڵ��б�
	vector<Move> untriedMoves;  // δ���Ե��߷�
	int visits;  // ��ǰ�ڵ㱻���ʴ���
	double value;  // ��ǰ�ڵ���ۼƼ�ֵ������ UCB��

	// ���캯��
	Node(const State& st, Node* p = nullptr, Move m = Move{ -1,-1,-1,-1,-1,-1 })
		: state(st), move(m), parent(p), visits(0), value(0.0) {
		state.getMoves(untriedMoves);  // �������еĺϷ��߷�
	}

	// ����UCBѡ������ӽڵ�
	Node* selectChild() {
		Node* best = nullptr;
		double bestU = -1e18;
		// ���������ӽڵ㣬ѡ��UCBֵ�����ӽڵ�
		for (size_t i = 0; i < children.size(); ++i) {
			Node* c = children[i];
			double u = c->value / c->visits + 1.414 * sqrt(log(visits) / c->visits);  // UCB��ʽ
			if (u > bestU) {
				bestU = u;
				best = c;
			}
		}
		return best;  // ��������ӽڵ�
	}

	// ��δ���Ե��߷���ѡһ���������µ��ӽڵ�
	Node* expand() {
		Move m = untriedMoves.back();  // ѡ�����һ��δ���Ե��߷�
		untriedMoves.pop_back();  // �Ƴ����߷�
		State next = state;
		next.doMove(m);  // ��������
		Node* child = new Node(next, this, m);  // �������ӽڵ�
		children.push_back(child);  // ���ӽڵ���ӵ��ӽڵ��б�
		return child;
	}

	// �ش�ģ���������µ�ǰ�ڵ��ֵ
	void backup(double result) {
		visits++;  // ���ʴ�����1
		value += result;  // �ۼ�ģ����
		if (parent) parent->backup(-result);  // ������¸��ڵ�Ľ��
	}
};

int main() {
	srand(time(NULL));  // ��ʼ�������������
	int x0, y0, x1, y1, x2, y2;

	// ��ʼ������״̬
	gridInfo[0][2] = grid_black;
	gridInfo[2][0] = grid_black;
	gridInfo[5][0] = grid_black;
	gridInfo[7][2] = grid_black;
	gridInfo[0][5] = grid_white;
	gridInfo[2][7] = grid_white;
	gridInfo[5][7] = grid_white;
	gridInfo[7][5] = grid_white;

	// ����غ���
	int turnID;
	cin >> turnID;

	// ���迪ʼʱ�ǰ׷�
	currBotColor = grid_white;
	for (int i = 0; i < turnID; ++i) {
		cin >> x0 >> y0 >> x1 >> y1 >> x2 >> y2;
		if (x0 == -1) currBotColor = grid_black;  // ����Է�����
		else ProcStep(x0, y0, x1, y1, x2, y2, -currBotColor, false);  // ģ��Է�����

		if (i < turnID - 1) {
			cin >> x0 >> y0 >> x1 >> y1 >> x2 >> y2;
			if (x0 >= 0) ProcStep(x0, y0, x1, y1, x2, y2, currBotColor, false);  // ģ�Ȿ������
		}
	}

	// ��ʼ�����ڵ�ľ���
	State rootState;
	for (int i = 0; i < GRIDSIZE; ++i)
		for (int j = 0; j < GRIDSIZE; ++j)
			rootState.board[i][j] = gridInfo[i][j];
	rootState.player = currBotColor;

	// �������ڵ�
	Node* root = new Node(rootState);
	auto start = high_resolution_clock::now();  // ��¼��ʼʱ��
	double elapsed = 0.0;

	// MCTS��ѭ��
	while (elapsed < 0.9) {  // ����ʱ��Ϊ0.9��
		Node* node = root;

		// ѡ�����չ�׶�
		while (node->untriedMoves.empty() && !node->children.empty() && !node->state.isTerminal()) {
			node = node->selectChild();  // ѡ��UCBֵ�����ӽڵ�
		}

		// ��չһ���ӽڵ�
		if (!node->untriedMoves.empty()) {
			node = node->expand();
		}

		// ���ģ��
		int result = node->state.rollout(currBotColor);
		node->backup(result);  // �ش�ģ����

		// ��������ʱ��
		auto now = high_resolution_clock::now();
		elapsed = duration<double>(now - start).count();
	}

	// ѡ����ʴ��������ӽڵ�
	Node* bestChild = nullptr;
	int maxVisits = -1;
	for (size_t i = 0; i < root->children.size(); ++i) {
		if (root->children[i]->visits > maxVisits) {
			maxVisits = root->children[i]->visits;
			bestChild = root->children[i];
		}
	}

	// ��������߷�
	if (bestChild) {
		Move m = bestChild->move;
		cout << m.x0 << " " << m.y0 << " " << m.x1 << " " << m.y1 << " " << m.x2 << " " << m.y2 << endl;
	}
	else {
		cout << "-1 -1 -1 -1 -1 -1" << endl;  // ���û�п��е��߷�
	}

	return 0;
}