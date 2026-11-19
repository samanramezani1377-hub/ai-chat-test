package com.samanramezani.aichattest.screen

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ScreenQuestionClassifierTest {
    @Test fun recognizesDirectPersianScreenQuestions() {
        assertTrue(ScreenQuestionClassifier.isScreenQuestion("این صفحه چی می‌گه؟"))
        assertTrue(ScreenQuestionClassifier.isScreenQuestion("این دکمه برای چیه؟"))
    }

    @Test fun recognizesPersianDeicticFollowUps() {
        assertTrue(ScreenQuestionClassifier.isScreenQuestion("اینو چطور حل کنم؟"))
        assertTrue(ScreenQuestionClassifier.isScreenQuestion("این رو انتخاب کنم؟"))
    }

    @Test fun recognizesEnglishScreenQuestions() {
        assertTrue(ScreenQuestionClassifier.isScreenQuestion("What should I click on the screen?"))
    }

    @Test fun doesNotRouteGenericQuestionsToScreenVision() {
        assertFalse(ScreenQuestionClassifier.isScreenQuestion("سلام، حالت چطوره؟"))
        assertFalse(ScreenQuestionClassifier.isScreenQuestion("برای خرید گوشی چی پیشنهاد می‌کنی؟"))
    }
}
