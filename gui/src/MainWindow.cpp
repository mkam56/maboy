#include "../include/MainWindow.h"

#include "../include/AnimatedButton.h"
#include "../include/ErrorPopup.h"

#include <QVBoxLayout>
#include <QWidget>
#include <QLabel>
#include <QFileInfo>
#include <QApplication>
#include <QThread>
#include <QDebug>
#include <vector>
#include <iostream>
#include <unistd.h>

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent)
{
	qRegisterMetaType<ValidationResult>("ValidationResult");
	
	auto* centralWidget = new QWidget(this);
	setCentralWidget(centralWidget);

	auto* qv_box_layout = new QVBoxLayout(centralWidget);

	fileDropPanel = new FileDropPanel(this);
	qv_box_layout->addWidget(fileDropPanel);

	resultLabel = new QLabel("No document loaded", this);
	resultLabel->setAlignment(Qt::AlignCenter);
	resultLabel->setStyleSheet("font-size: 18px; padding: 20px; color: #555;");
	resultLabel->setWordWrap(true);
	qv_box_layout->addWidget(resultLabel);

	progressBar = new QProgressBar(this);
	progressBar->setRange(0, 100);
	progressBar->setValue(0);
	progressBar->setTextVisible(true);
	progressBar->setFormat("Processing: %p%");
	progressBar->setStyleSheet(
		"QProgressBar {"
		"    border: 2px solid #a285e2;"
		"    border-radius: 5px;"
		"    text-align: center;"
		"    background-color: #f8f0ff;"
		"    min-height: 30px;"
		"    font-size: 14px;"
		"}"
		"QProgressBar::chunk {"
		"    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #a285e2, stop:1 #c8b3f0);"
		"    border-radius: 3px;"
		"}"
	);
	progressBar->setVisible(false);
	qv_box_layout->addWidget(progressBar);

	progressTimer = new QTimer(this);
	connect(progressTimer, &QTimer::timeout, this, &MainWindow::updateProgressBar);
	targetProgress = 0;
	slowProgress = false;

	validateButton = new AnimatedButton("Validate Document", this);
	validateButton->setEnabled(false);
	qv_box_layout->addWidget(validateButton);

	connect(fileDropPanel, &FileDropPanel::fileSelected, this, &MainWindow::updateValidateButtonState);
	connect(validateButton, &QPushButton::clicked, this, &MainWindow::validateDocument);

	errorSound = new QSoundEffect(this);
	errorSound->setSource(QUrl("qrc:/sounds/error.wav"));
	errorSound->setVolume(0.3f);

	successSound = new QSoundEffect(this);
	successSound->setSource(QUrl("qrc:/sounds/success.wav"));
	successSound->setVolume(0.3f);

	validator = new DocumentValidator();
	
	workerThread = new QThread(this);
	worker = new ValidationWorker(validator);
	worker->moveToThread(workerThread);
	
	connect(this, &MainWindow::destroyed, workerThread, &QThread::quit);
	connect(workerThread, &QThread::finished, worker, &QObject::deleteLater);
	connect(this, &MainWindow::startValidation, worker, &ValidationWorker::doValidation);
	connect(worker, &ValidationWorker::validationComplete, this, &MainWindow::validationFinished);
	
	workerThread->start();
	
	char cwd[1024];
	getcwd(cwd, sizeof(cwd));
	
	std::vector<std::string> possible_roots = {
		"/Users/mehkam/CLionProjects/maboy",
		"/Users/mehkam/doc_val_ml",
		std::string(cwd)
	};
	
	bool models_loaded = false;
	std::string loaded_root;
	
	for (const auto& root : possible_roots) {
		if (validator->loadModels(root)) {
			models_loaded = true;
			loaded_root = root;
			break;
		}
	}
	
	if (models_loaded) {
		QString msg = QString("Models loaded successfully!\n(%1)\n\nReady for validation.")
			.arg(QString::fromStdString(loaded_root));
		resultLabel->setText(msg);
		resultLabel->setStyleSheet("font-size: 14px; padding: 20px; color: #28a745;");
	} else {
		QString msg = "Models not found!\n\nRequired files:\nvalidator.py\nModel files (.pt, .pth, .pkl)";
		resultLabel->setText(msg);
		resultLabel->setStyleSheet("font-size: 12px; padding: 20px; color: #ff6b6b;");
	}

	setWindowTitle("Document Validator - LibTorch");
	resize(600, 450);
}

MainWindow::~MainWindow()
{
	if (workerThread) {
		workerThread->quit();
		workerThread->wait();
	}
	delete validateButton;
	delete resultLabel;
	delete validator;
}

void MainWindow::validateDocument()
{
	const std::string file_path = fileDropPanel->selectedFilePath().toStdString();
	
	progressBar->setVisible(true);
	progressBar->setValue(0);
	targetProgress = 5;
	progressTimer->start(80);
	resultLabel->setText("Initializing validation...");
	resultLabel->setStyleSheet("font-size: 14px; padding: 20px; color: #007bff;");
	validateButton->setEnabled(false);
	
	QTimer::singleShot(200, [this, file_path]() {
		targetProgress = 15;
		resultLabel->setText("Loading models...");
	});
	
	QTimer::singleShot(350, [this, file_path]() {
		targetProgress = 30;
		resultLabel->setText("Processing image...");
	});
	
	QTimer::singleShot(450, [this, file_path]() {
		targetProgress = 50;
		resultLabel->setText("Running AI analysis...");
		slowProgress = true;
		targetProgress = 88;
		qDebug() << "[Main Thread] Emitting startValidation signal...";
		emit startValidation(QString::fromStdString(file_path));
		qDebug() << "[Main Thread] Signal emitted.";
	});
}

void MainWindow::validationFinished(const ValidationResult& result)
{
	qDebug() << "[Main Thread] Validation finished callback received!";
	slowProgress = false;
	targetProgress = 90;
	resultLabel->setText("Finalizing results...");
	
	QTimer::singleShot(100, [this, result]() {
		validateButton->setEnabled(true);
		
		if (result.detailed_message.empty()) {
			progressBar->setVisible(false);
			progressTimer->stop();
			resultLabel->setText("Error: Validation failed");
			resultLabel->setStyleSheet("font-size: 14px; padding: 20px; color: #dc3545;");
			errorSound->play();
			auto* popup = new ErrorPopup("Validation Error", false, this);
			popup->exec();
			return;
		}
		
		float avg_confidence = (result.realism_model_1.confidence + 
		                       result.realism_model_2.confidence + 
		                       result.realism_model_3.confidence + 
		                       result.ocr_model.confidence) / 4.0f * 100.0f;
		
		QString summary = QString("REALISM CHECK:\n");
		summary += QString("Model 1: %1 (%2%)\nModel 2: %3 (%4%)\nModel 3: %5 (%6%)\n\n")
			.arg(result.realism_model_1.is_valid ? "VALID" : "INVALID")
			.arg(result.realism_model_1.confidence * 100.0f, 0, 'f', 1)
			.arg(result.realism_model_2.is_valid ? "VALID" : "INVALID")
			.arg(result.realism_model_2.confidence * 100.0f, 0, 'f', 1)
			.arg(result.realism_model_3.is_valid ? "VALID" : "INVALID")
			.arg(result.realism_model_3.confidence * 100.0f, 0, 'f', 1);
		
		summary += QString("OCR CHECK:\nFields: %1 (%2%)\n\n")
			.arg(result.ocr_valid ? "CORRECT" : "INCORRECT")
			.arg(result.ocr_model.confidence * 100.0f, 0, 'f', 1);
		
		summary += QString("VERDICT: %1")
			.arg(result.final_verdict ? "VALID" : "INVALID");

		targetProgress = 100;
		
		QTimer::singleShot(150, [this, result, summary]() {
			progressTimer->stop();
			progressBar->setValue(100);
			
			QTimer::singleShot(300, [this, result, summary]() {
				progressBar->setVisible(false);
				
				if (result.final_verdict) {
					resultLabel->setText("DOCUMENT VALID");
					resultLabel->setStyleSheet("font-size: 24px; padding: 20px; color: #28a745; font-weight: bold;");
					successSound->play();
					auto* popup = new ErrorPopup(summary.toStdString().c_str(), true, this);
					popup->exec();
				} else {
					resultLabel->setText("DOCUMENT INVALID");
					resultLabel->setStyleSheet("font-size: 24px; padding: 20px; color: #dc3545; font-weight: bold;");
					errorSound->play();
					auto* popup = new ErrorPopup(summary.toStdString().c_str(), false, this);
					popup->exec();
				}
			});
		});
	});
}

void MainWindow::updateProgressBar()
{
	int current = progressBar->value();
	if (slowProgress && current < targetProgress) {
		if (current < 85) {
			progressBar->setValue(current + 1);
		}
	} else if (current < targetProgress) {
		int step = (targetProgress - current) / 5;
		if (step < 1) step = 1;
		progressBar->setValue(current + step);
	} else if (current > targetProgress) {
		progressBar->setValue(targetProgress);
	}
}

void MainWindow::updateValidateButtonState() const
{
	if (!fileDropPanel->selectedFilePath().isEmpty())
	{
		validateButton->setEnabled(true);
		resultLabel->setText("✓ Document loaded. Click 'Validate' to check.");
		resultLabel->setStyleSheet("font-size: 16px; padding: 20px; color: #555;");
	}
}
