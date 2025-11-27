#include "../include/MainWindow.h"

#include <QApplication>
#include <QFile>
#include <QFontDatabase>
#include <QGraphicsDropShadowEffect>

int main(int argc, char *argv[])
{
	QApplication app(argc, argv);
	QFile styleFile(":/colors.qss");
	if (styleFile.open(QFile::ReadOnly))
	{
		const QString styleSheet = QLatin1String(styleFile.readAll());
		app.setStyleSheet(styleSheet);
	}
	const int id = QFontDatabase::addApplicationFont(":/fronts/IBMPlexMono.ttf");
	const QString family = QFontDatabase::applicationFontFamilies(id).at(0);
	QApplication::setFont(QFont(family));
	MainWindow window;
	window.show();
	return app.exec();
}
