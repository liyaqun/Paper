import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QComboBox, QPushButton,
                             QLineEdit, QGroupBox, QFormLayout, QMessageBox,
                             QTableWidget, QTableWidgetItem, QHeaderView,
                             QSplitter)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor
import matplotlib
from model import Net

# 4163 名学生、17746 个练习题和 123个知识概念，
exer_n = 17746
knowledge_n = 123
student_n = 4163

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class KnowledgeVisualizationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("知识点掌握度分析系统")
        self.setGeometry(100, 100, 1400, 900)

        # 加载模型
        self.model = Net(student_n, exer_n, knowledge_n)
        self.load_model('model/model_epoch4')  # 使用训练好的模型

        # 主布局
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # 左侧控制面板
        control_panel = QGroupBox("控制面板")
        control_layout = QVBoxLayout(control_panel)
        control_layout.setSpacing(15)

        # 学生选择
        self.student_selector = QComboBox()
        self.student_selector.setFont(QFont("Microsoft YaHei", 10))
        for i in range(1, student_n + 1):
            self.student_selector.addItem(f"学生 {i}", i - 1)

        # 知识点选择
        self.knowledge_selector = QComboBox()
        self.knowledge_selector.setFont(QFont("Microsoft YaHei", 10))
        for i in range(1, knowledge_n + 1):
            self.knowledge_selector.addItem(f"知识点 {i}", i - 1)

        # 题目选择
        self.exercise_selector = QComboBox()
        self.exercise_selector.setFont(QFont("Microsoft YaHei", 10))
        for i in range(1, exer_n + 1):
            self.exercise_selector.addItem(f"题目 {i}", i - 1)

        # 预测按钮
        # predict_btn = QPushButton("预测答题正确率")
        # predict_btn.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        # predict_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        # predict_btn.clicked.connect(self.predict_performance)

        # 表单布局
        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignRight)
        form_layout.addRow("选择学生:", self.student_selector)
        form_layout.addRow("选择知识点:", self.knowledge_selector)
        # form_layout.addRow("选择题目:", self.exercise_selector)
        # form_layout.addRow(predict_btn)

        # 预测结果展示
        self.prediction_result = QLabel("预测结果将显示在这里")
        self.prediction_result.setFont(QFont("Microsoft YaHei", 10))
        self.prediction_result.setAlignment(Qt.AlignCenter)
        self.prediction_result.setStyleSheet("""
            font-size: 14px; 
            font-weight: bold;
            background-color: #f0f8ff;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #ccc;
        """)

        # 知识点掌握程度表格
        self.knowledge_table = QTableWidget()
        self.knowledge_table.setColumnCount(3)
        self.knowledge_table.setHorizontalHeaderLabels(["知识点", "掌握程度", "平均难度"])
        self.knowledge_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.knowledge_table.setFont(QFont("Microsoft YaHei", 9))
        self.knowledge_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.knowledge_table.setRowCount(knowledge_n)

        # 添加到控制面板
        control_layout.addLayout(form_layout)
        control_layout.addWidget(self.prediction_result)
        control_layout.addWidget(QLabel("知识点掌握程度:"))
        control_layout.addWidget(self.knowledge_table)

        # 右侧可视化区域
        visualization_panel = QGroupBox("可视化展示")
        visualization_layout = QVBoxLayout(visualization_panel)

        # 创建图表
        self.figure, self.ax = plt.subplots(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        visualization_layout.addWidget(self.canvas)

        # 使用分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(control_panel)
        splitter.addWidget(visualization_panel)
        splitter.setSizes([400, 1000])

        main_layout.addWidget(splitter)
        self.setCentralWidget(main_widget)

        # 初始显示第一个学生的掌握情况
        self.student_selector.currentIndexChanged.connect(self.update_visualization)
        self.knowledge_selector.currentIndexChanged.connect(self.update_visualization)
        self.update_visualization()

    def load_model(self, model_path):
        """加载训练好的模型"""
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.model.eval()
            print("模型加载成功!")

            # 计算每个知识点的平均难度
            self.avg_knowledge_difficulty = np.zeros(knowledge_n)
            for exer_id in range(exer_n):
                k_difficulty, _ = self.get_exercise_params(exer_id)
                self.avg_knowledge_difficulty += k_difficulty
            self.avg_knowledge_difficulty /= exer_n
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法加载模型: {str(e)}")

    def get_student_knowledge_status(self, stu_id):
        """获取学生的知识点掌握情况"""
        with torch.no_grad():
            knowledge_status = self.model.get_knowledge_status(torch.LongTensor([stu_id]))
            return knowledge_status.numpy().flatten()

    def get_exercise_params(self, exer_id):
        """获取题目的参数"""
        with torch.no_grad():
            # 难度  区分度
            k_difficulty, e_discrimination = self.model.get_exer_params(torch.LongTensor([exer_id]))
            return k_difficulty.numpy().flatten(), e_discrimination.numpy()[0][0]

    def predict_performance(self):
        """预测学生在特定题目上的表现"""
        stu_id = self.student_selector.currentData()
        exer_id = self.exercise_selector.currentData()

        # 获取题目相关的知识点嵌入（这里简化处理）
        # kn_emb = torch.ones(1, knowledge_n)

        # with torch.no_grad():
        #     prediction = self.model(
        #         torch.LongTensor([stu_id]),
        #         torch.LongTensor([exer_id]),
        #         kn_emb
        #     ).item()

        # 获取题目参数
        k_difficulty, e_discrimination = self.get_exercise_params(exer_id)

        # 显示预测结果
        result_text = (
            # 不显示在某个题目上的预测结果了:因为这个题目的知识点不确定
            # f"学生 {stu_id + 1} 在题目 {exer_id + 1} 上的预测结果:\n\n"
            # f"预测正确率: <b>{prediction * 100:.2f}%</b>\n\n"
            f"题目参数:\n"
            f"  区分度: <b>{e_discrimination:.4f}</b>\n"
            f"  知识点难度:"
        )

        # 添加相关知识点难度
        for i, diff in enumerate(k_difficulty):
            if diff > 0.01:  # 只显示难度较高的知识点
                result_text += f"\n    知识点 {i + 1}: {diff:.4f}"

        self.prediction_result.setText(result_text)

        # 在图表中突出显示当前知识点
        self.update_visualization(highlight_knowledge=self.knowledge_selector.currentData() - 1)

    def update_visualization(self, highlight_knowledge=None):
        """更新可视化图表"""
        stu_id = self.student_selector.currentData()
        knowledge_status = self.get_student_knowledge_status(stu_id)

        # 清空图表
        self.ax.clear()

        # 创建柱状图
        indices = np.arange(len(knowledge_status))

        # 绘制掌握程度
        mastery_bars = self.ax.bar(indices - 0.2, knowledge_status, width=0.4, color='skyblue', label='掌握程度')

        # 绘制知识点平均难度
        difficulty_bars = self.ax.bar(indices + 0.2, self.avg_knowledge_difficulty, width=0.4, color='lightgreen',
                                      label='平均难度')

        # 突出显示选中的知识点
        if highlight_knowledge is not None:
            mastery_bars[highlight_knowledge].set_color('red')
            difficulty_bars[highlight_knowledge].set_color('darkred')

            # 添加文本标注
            mastery = knowledge_status[highlight_knowledge]
            difficulty = self.avg_knowledge_difficulty[highlight_knowledge]
            self.ax.annotate(f'掌握: {mastery:.4f}\n难度: {difficulty:.4f}',
                             xy=(highlight_knowledge, max(mastery, difficulty) + 0.05),
                             xytext=(highlight_knowledge, max(mastery, difficulty) + 0.15),
                             arrowprops=dict(facecolor='black', shrink=0.05),
                             ha='center', va='bottom', fontsize=10)

        # 设置图表属性
        self.ax.set_title(f"学生 {stu_id + 1} 的知识点掌握情况", fontsize=16)
        self.ax.set_xlabel("知识点编号", fontsize=12)
        self.ax.set_ylabel("程度值 (0-1)", fontsize=12)
        self.ax.set_ylim(0, 1)
        self.ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        self.ax.legend()

        # 只显示每5个知识点的标签
        xticks = indices[::5]
        self.ax.set_xticks(xticks)
        self.ax.set_xticklabels([f"{i + 1}" for i in xticks])

        # 添加平均掌握度参考线
        avg_mastery = np.mean(knowledge_status)
        self.ax.axhline(avg_mastery, color='r', linestyle='--', alpha=0.7)
        self.ax.text(len(knowledge_status) - 5, avg_mastery + 0.03,
                     f'平均掌握度: {avg_mastery:.2f}',
                     color='r', fontsize=12)

        # 重新绘制
        self.canvas.draw()

        # 更新知识点表格
        self.update_knowledge_table(knowledge_status)

    def update_knowledge_table(self, knowledge_status):
        """更新知识点表格数据"""
        for i in range(knowledge_n):
            # 知识点编号
            item_id = QTableWidgetItem(f"知识点 {i + 1}")

            # 掌握程度
            mastery = knowledge_status[i]
            item_mastery = QTableWidgetItem(f"{mastery:.4f}")

            # 设置颜色
            # 掌握程度
            if mastery < 0.4:
                item_mastery.setBackground(QColor(255, 200, 200))  # 红色
            elif mastery < 0.6:
                item_mastery.setBackground(QColor(255, 255, 200))  # 黄色
            else:
                item_mastery.setBackground(QColor(200, 255, 200))  # 绿色

            # 平均难度
            difficulty = self.avg_knowledge_difficulty[i]
            item_difficulty = QTableWidgetItem(f"{difficulty:.4f}")

            # 设置颜色
            if difficulty > 0.7:
                item_difficulty.setBackground(QColor(255, 200, 200))  # 红色
            elif difficulty > 0.4:
                item_difficulty.setBackground(QColor(255, 255, 200))  # 黄色
            else:
                item_difficulty.setBackground(QColor(200, 255, 200))  # 绿色

            self.knowledge_table.setItem(i, 0, item_id)
            self.knowledge_table.setItem(i, 1, item_mastery)
            self.knowledge_table.setItem(i, 2, item_difficulty)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置全局字体
    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)

    window = KnowledgeVisualizationApp()
    window.show()
    sys.exit(app.exec_())
