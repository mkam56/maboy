#pragma once

#include "../include/AnimatedButton.h"

#include <QDragLeaveEvent>
#include <QFileDialog>
#include <QGraphicsDropShadowEffect>
#include <QHBoxLayout>
#include <QLabel>
#include <QMimeData>
#include <QPushButton>
#include <QSoundEffect>

class QLabel;
class QPushButton;

class FileDropPanel final : public QWidget
{
	Q_OBJECT

  public:
	explicit FileDropPanel(QWidget* parent = nullptr);

	QString selectedFilePath() const;

  signals:
	void fileSelected(const QString& path);

  protected:
	void dragEnterEvent(QDragEnterEvent* event) override;
	void dropEvent(QDropEvent* event) override;
	void dragLeaveEvent(QDragLeaveEvent* event) override;

  private slots:
	void onBrowseClicked();

  private:
	void updateFileLabel(const QString& path);

	QLabel* fileLabel;
	QPushButton* browseBtn;
	QString filePath;
	QPropertyAnimation* animation;
	QGraphicsDropShadowEffect* shadow;
	QSoundEffect* dropSound;
};
