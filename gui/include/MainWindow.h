#pragma once

#include "FileDropPanel.h"
#include "../../include/DocumentValidator.h"

#include <QCheckBox>
#include <QGraphicsView>
#include <QMainWindow>
#include <QProgressBar>
#include <QTimer>
#include <QThread>
#include <QDebug>

class ValidationWorker : public QObject
{
	Q_OBJECT
public:
	ValidationWorker(DocumentValidator* validator) : validator_(validator) {}
	
public slots:
	void doValidation(QString filePath) {
		qDebug() << "[Worker Thread] Starting validation for:" << filePath;
		ValidationResult result = validator_->validate(filePath.toStdString());
		qDebug() << "[Worker Thread] Validation complete, emitting result...";
		emit validationComplete(result);
	}
	
signals:
	void validationComplete(const ValidationResult& result);
	
private:
	DocumentValidator* validator_;
};

class MainWindow final : public QMainWindow
{
  public:
	explicit MainWindow(QWidget* parent = nullptr);
	~MainWindow() override;

  private slots:
	void validateDocument();
	void updateValidateButtonState() const;
	void updateProgressBar();
	void validationFinished(const ValidationResult& result);

  signals:
	void startValidation(QString filePath);

  private:
	Q_OBJECT
	QPushButton* validateButton;
	FileDropPanel* fileDropPanel;
	QLabel* resultLabel;
	QProgressBar* progressBar;
	QTimer* progressTimer;
	int targetProgress;
	bool slowProgress;
	QSoundEffect* successSound;
	QSoundEffect* errorSound;
	
	DocumentValidator* validator;
	QThread* workerThread;
	ValidationWorker* worker;
};
