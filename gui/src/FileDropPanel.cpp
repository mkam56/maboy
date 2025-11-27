#include "../include/FileDropPanel.h"

#include <QGraphicsDropShadowEffect>

FileDropPanel::FileDropPanel(QWidget* parent) : QWidget(parent), filePath("")
{
	setAcceptDrops(true);

	fileLabel = new QLabel("The file is not selected", this);

	fileLabel->setAlignment(Qt::AlignCenter);
	fileLabel->setStyleSheet(
		"border: 1px dashed gray; padding: 10px; color: #3b3b3b; font-size: 16px; font-weight: "
		"bold;");

	browseBtn = new AnimatedButton("Select a file", this);
	connect(browseBtn, &QPushButton::clicked, this, &FileDropPanel::onBrowseClicked);

	setStyleSheet(R"(
        QWidget {
            background-color: #f8f0ff;
            border: 2px dashed #a285e2;
            border-radius: 12px;
        }
    )");

	shadow = new QGraphicsDropShadowEffect(this);
	shadow->setBlurRadius(15);
	shadow->setOffset(0, 5);
	shadow->setColor(QColor(0, 0, 0, 100));
	setGraphicsEffect(shadow);

	animation = new QPropertyAnimation(this, "geometry");
	animation->setDuration(150);

	dropSound = new QSoundEffect(this);
	dropSound->setSource(QUrl("qrc:/sounds/file_drop.wav"));
	dropSound->setVolume(0.25f);

	auto* layout = new QVBoxLayout(this);
	layout->addWidget(fileLabel);
	layout->addWidget(browseBtn);
	setLayout(layout);
}

QString FileDropPanel::selectedFilePath() const
{
	return filePath;
}

void FileDropPanel::dragEnterEvent(QDragEnterEvent* event)
{
	if (event->mimeData()->hasUrls() && event->mimeData()->urls().size() == 1)
	{
		event->acceptProposedAction();
		const QRect start = geometry();
		const auto end = QRect(start.x() - 5, start.y() - 5, start.width() + 10, start.height() + 10);

		animation->stop();
		animation->setStartValue(start);
		animation->setEndValue(end);
		animation->start();
	}
}

void FileDropPanel::dropEvent(QDropEvent* event)
{
	if (event->mimeData()->hasUrls())
	{
		if (const QList< QUrl > urls = event->mimeData()->urls(); !urls.isEmpty())
		{
			const QString path = urls.first().toLocalFile();
			updateFileLabel(path);
			emit fileSelected(path);
		}
	}
}

void FileDropPanel::dragLeaveEvent(QDragLeaveEvent* event)
{
	const QRect end = geometry();
	const auto start = QRect(end.x() + 5, end.y() + 5, end.width() - 10, end.height() - 10);

	animation->stop();
	animation->setStartValue(end);
	animation->setEndValue(start);
	animation->start();
}

void FileDropPanel::onBrowseClicked()
{
	if (const QString path = QFileDialog::getOpenFileName(this, "Select a file"); !path.isEmpty())
	{
		updateFileLabel(path);
		emit fileSelected(path);
	}
}

void FileDropPanel::updateFileLabel(const QString& path)
{
	dropSound->play();
	filePath = path;
	const QFileInfo fileInfo(path);
	fileLabel->setText(fileInfo.fileName());
}
